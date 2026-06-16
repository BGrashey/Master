import math
import numpy as np
import dask
import dask.array as da
from itertools import product
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.wcs import WCS
import zarr

# ==========================================
# 1. UNION-FIND & CHUNK GEOMETRY HELPERS
# ==========================================
class UnionFind:
    __slots__ = ("parent", "rank")
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        else:
            self.parent.setdefault(x, x)
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb: return
        rka = self.rank.get(ra, 0)
        rkb = self.rank.get(rb, 0)
        if rka < rkb: self.parent[ra] = rb
        elif rka > rkb: self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = rka + 1

def _prefix_offsets(chunks_1d: Tuple[int, ...]) -> np.ndarray:
    off = np.zeros(len(chunks_1d) + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.array(chunks_1d, dtype=np.int64))
    return off

def _chunk_origin_from_offsets(offx, offy, offz, block_id: Tuple[int, int, int]) -> Tuple[int, int, int]:
    i, j, k = block_id
    return int(offx[i]), int(offy[j]), int(offz[k])

def _linear_chunk_id(block_id: Tuple[int, int, int], numblocks: Tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index(block_id, numblocks))

def _group_minmax(values: np.ndarray, group_idx: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0 or n_groups == 0:
        return (np.empty((0,), dtype=values.dtype), np.empty((0,), dtype=values.dtype))
    order = np.argsort(group_idx, kind="mergesort")
    gi = group_idx[order]
    vv = values[order]
    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    gids = gi[starts]
    ends = np.r_[starts[1:], vv.size]
    vmin = np.full((n_groups,), np.iinfo(vv.dtype).max, dtype=vv.dtype)
    vmax = np.full((n_groups,), np.iinfo(vv.dtype).min, dtype=vv.dtype)
    for g, s, e in zip(gids, starts, ends):
        seg = vv[s:e]
        vmin[g] = seg.min()
        vmax[g] = seg.max()
    return vmin, vmax

def _map_ids_with_mapping(ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if ids.size == 0 or keys.size == 0: return ids.astype(np.int64, copy=False)
    idx = np.searchsorted(keys, ids)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == ids[valid])
    out = ids.copy()
    out[hit] = values[idx[hit]]
    return out

# ==========================================
# 2. CORE FOF PIPELINE (MINIMAL CONFIG)
# ==========================================
@dataclass
class ChunkResult:
    labels: np.ndarray
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]]
    n_local: int
    chunk_id: int
    catalog_local: Dict[str, np.ndarray]

def _fof_labels_for_positions(coords_local: np.ndarray, linking_length: float, kdtree_workers: int = -1) -> np.ndarray:
    n = coords_local.shape[0]
    if n == 0: return np.zeros((0,), dtype=np.int32)
    tree = cKDTree(coords_local)
    visited = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int32)
    cur_label = 0
    for p in range(n):
        if visited[p]: continue
        cur_label += 1
        stack = [p]
        visited[p] = True
        labels[p] = cur_label
        while stack:
            cur = stack.pop()
            neigh = tree.query_ball_point(coords_local[cur], r=linking_length, workers=kdtree_workers)
            for q in neigh:
                if not visited[q]:
                    visited[q] = True
                    labels[q] = cur_label
                    stack.append(q)
    return labels

def _extract_face_shell(coords_local: np.ndarray, point_labels: np.ndarray, chunk_shape: Tuple[int, int, int], origin: Tuple[int, int, int], shell: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    faces = {f: (np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)) for f in ("x0", "x1", "y0", "y1", "z0", "z1")}
    if coords_local.shape[0] == 0: return faces
    sh = int(shell)
    sx, sy, sz = chunk_shape
    x, y, z = coords_local[:, 0], coords_local[:, 1], coords_local[:, 2]
    masks = {"x0": x < sh, "x1": x >= (sx - sh), "y0": y < sh, "y1": y >= (sy - sh), "z0": z < sh, "z1": z >= (sz - sh)}
    ox, oy, oz = origin
    for key, m in masks.items():
        c = coords_local[m].astype(np.int32, copy=False)
        if c.shape[0] == 0: continue
        g = c.copy()
        g[:, 0] += ox; g[:, 1] += oy; g[:, 2] += oz
        faces[key] = (g, point_labels[m].astype(np.int32, copy=False))
    return faces

def _catalog_for_chunk_points(coords_local: np.ndarray, point_labels: np.ndarray, origin: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    npts = coords_local.shape[0]
    if npts == 0:
        return {
            "local_label": np.empty((0,), dtype=np.int64), "n_spax": np.empty((0,), dtype=np.int64),
            "xmin": np.empty((0,), dtype=np.int64), "xmax": np.empty((0,), dtype=np.int64),
            "ymin": np.empty((0,), dtype=np.int64), "ymax": np.empty((0,), dtype=np.int64),
            "zmin": np.empty((0,), dtype=np.int64), "zmax": np.empty((0,), dtype=np.int64),
        }
    n_local = int(point_labels.max())
    group_idx = point_labels.astype(np.int64) - 1
    oz, oy, ox = origin
    gz = coords_local[:, 0].astype(np.int64) + oz
    gy = coords_local[:, 1].astype(np.int64) + oy
    gx = coords_local[:, 2].astype(np.int64) + ox

    n_spax = np.bincount(group_idx, minlength=n_local).astype(np.int64)
    xmin, xmax = _group_minmax(gx, group_idx, n_local)
    ymin, ymax = _group_minmax(gy, group_idx, n_local)
    zmin, zmax = _group_minmax(gz, group_idx, n_local)

    return {
        "local_label": np.arange(n_local, dtype=np.int64) + 1, "n_spax": n_spax,
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax
    }

def _process_one_chunk(block_bool: np.ndarray, origin: Tuple[int, int, int], chunk_id: int, linking_length: float, shell: int, kdtree_workers: int = -1) -> ChunkResult:
    coords = np.argwhere(block_bool)
    point_labels = _fof_labels_for_positions(coords, linking_length, kdtree_workers=kdtree_workers)
    labels = np.zeros(block_bool.shape, dtype=np.int32)
    if coords.shape[0] > 0:
        labels[coords[:, 0], coords[:, 1], coords[:, 2]] = point_labels
    n_local = int(point_labels.max()) if point_labels.size else 0
    faces_local = _extract_face_shell(coords, point_labels, block_bool.shape, origin, shell)
    faces_global = {}
    for face, (gcoords, labs_local) in faces_local.items():
        if gcoords.shape[0] == 0:
            faces_global[face] = (gcoords, np.empty((0,), dtype=np.int64))
            continue
        gids = (np.int64(chunk_id) << np.int64(32)) | (labs_local.astype(np.int64) & np.int64(0xFFFFFFFF))
        faces_global[face] = (gcoords, gids.astype(np.int64, copy=False))
    cat = _catalog_for_chunk_points(coords, point_labels, origin)
    ll = cat["local_label"].astype(np.int64, copy=False)
    gids_local = (np.int64(chunk_id) << np.int64(32)) | (ll & np.int64(0xFFFFFFFF)) if ll.size else np.empty((0,), dtype=np.int64)
    cat["gid"] = gids_local
    cat.pop("local_label", None)
    return ChunkResult(labels=labels, faces=faces_global, n_local=n_local, chunk_id=chunk_id, catalog_local=cat)

def _edges_between_faces(faceA: Tuple[np.ndarray, np.ndarray], faceB: Tuple[np.ndarray, np.ndarray], linking_length: float, kdtree_workers: int = -1) -> np.ndarray:
    coordsA, gidsA = faceA
    coordsB, gidsB = faceB
    if coordsA.shape[0] == 0 or coordsB.shape[0] == 0: return np.empty((0, 2), dtype=np.int64)
    treeB = cKDTree(coordsB.astype(np.float32, copy=False))
    edges = []
    for i in range(coordsA.shape[0]):
        neigh = treeB.query_ball_point(coordsA[i].astype(np.float32, copy=False), r=linking_length, workers=kdtree_workers)
        ga = int(gidsA[i])
        for j in neigh:
            gb = int(gidsB[j])
            if ga != gb:
                edges.append((ga, gb) if ga < gb else (gb, ga))
    if not edges: return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.array(edges, dtype=np.int64), axis=0)

def _concat_edges(*edge_arrays: np.ndarray) -> np.ndarray:
    nonempty = [e for e in edge_arrays if e is not None and e.size]
    if not nonempty: return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.vstack(nonempty), axis=0)

def _build_mapping_from_edges(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if edges.size == 0: return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))
    uf = UnionFind()
    for a, b in edges: uf.union(int(a), int(b))
    nodes = np.unique(edges.reshape(-1)).astype(np.int64)
    root_to_label, next_label = {}, 1
    vals = np.empty(nodes.shape[0], dtype=np.int64)
    for i, n in enumerate(nodes):
        r = uf.find(int(n))
        lab = root_to_label.get(r)
        if lab is None:
            lab = next_label
            root_to_label[r] = lab
            next_label += 1
        vals[i] = lab
    order = np.argsort(nodes)
    return nodes[order], vals[order]

def _relabel_chunk(chunk_res, keys, values):
    lab_local = chunk_res.labels
    out = np.zeros(lab_local.shape, dtype=np.int64)
    m = lab_local > 0
    if not np.any(m): return out
    packed = ((np.int64(chunk_res.chunk_id) << np.int64(32)) | (lab_local[m].astype(np.int64) & np.int64(0xFFFFFFFF)))
    if keys.size == 0:
        out[m] = packed
        return out
    idx = np.searchsorted(keys, packed)
    valid = idx < keys.size
    hit = np.zeros_like(valid, dtype=bool)
    hit[valid] = (keys[idx[valid]] == packed[valid])
    out_m = out[m]
    out_m[hit] = values[idx[hit]]
    out_m[~hit] = packed[~hit]
    out[m] = out_m
    return out

def _finalize_chunk_catalog(chunk_res: ChunkResult, keys: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
    cat = chunk_res.catalog_local
    out = {k: v for k, v in cat.items() if k != "gid"}
    out["id"] = _map_ids_with_mapping(cat["gid"], keys, values)
    return out

def _merge_catalog_dicts(cat_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = np.concatenate([c["id"] for c in cat_list if c["id"].size], axis=0)
    fields = ["id", "n_spax", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    if ids.size == 0: return {k: np.empty((0,), dtype=np.int64) for k in fields}
    
    n_spax = np.concatenate([c["n_spax"] for c in cat_list if c["id"].size])
    xmin = np.concatenate([c["xmin"] for c in cat_list if c["id"].size])
    xmax = np.concatenate([c["xmax"] for c in cat_list if c["id"].size])
    ymin = np.concatenate([c["ymin"] for c in cat_list if c["id"].size])
    ymax = np.concatenate([c["ymax"] for c in cat_list if c["id"].size])
    zmin = np.concatenate([c["zmin"] for c in cat_list if c["id"].size])
    zmax = np.concatenate([c["zmax"] for c in cat_list if c["id"].size])

    order = np.argsort(ids, kind="mergesort")
    ids_s = ids[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])

    return {
        "id": ids_s[starts],
        "n_spax": np.add.reduceat(n_spax[order], starts),
        "xmin": np.minimum.reduceat(xmin[order], starts), "xmax": np.maximum.reduceat(xmax[order], starts),
        "ymin": np.minimum.reduceat(ymin[order], starts), "ymax": np.maximum.reduceat(ymax[order], starts),
        "zmin": np.minimum.reduceat(zmin[order], starts), "zmax": np.maximum.reduceat(zmax[order], starts),
    }

# ==========================================
# 3. PUBLIC API & WCS EVALUATION
# ==========================================
def fof_minimal_zarr(arr_bool, linking_length: float, kdtree_workers: int = -1):
    darr = da.asarray(arr_bool)
    shell = int(math.ceil(float(linking_length)))
    chunks, numblocks = darr.chunks, darr.numblocks
    offx, offy, offz = _prefix_offsets(chunks[0]), _prefix_offsets(chunks[1]), _prefix_offsets(chunks[2])
    delayed_blocks_bool = darr.to_delayed().ravel()
    block_ids = list(product(range(numblocks[0]), range(numblocks[1]), range(numblocks[2])))

    chunk_results = []
    for idx, bid in enumerate(block_ids):
        origin = _chunk_origin_from_offsets(offx, offy, offz, bid)
        cid = _linear_chunk_id(bid, numblocks)
        chunk_results.append(dask.delayed(_process_one_chunk)(delayed_blocks_bool[idx], origin, cid, linking_length, shell, kdtree_workers))

    idx_of = {bid: i for i, bid in enumerate(block_ids)}
    neighbor_dirs, dir_faces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)], {(1, 0, 0): ("x1", "x0"), (0, 1, 0): ("y1", "y0"), (0, 0, 1): ("z1", "z0")}
    edge_tasks = []
    for bid in block_ids:
        for d in neighbor_dirs:
            nb = (bid[0] + d[0], bid[1] + d[1], bid[2] + d[2])
            if not ((0 <= nb[0] < numblocks[0]) and (0 <= nb[1] < numblocks[1]) and (0 <= nb[2] < numblocks[2])): continue
            fa, fb = dir_faces[d]
            edge_tasks.append(dask.delayed(lambda ra, rb, fa=fa, fb=fb: _edges_between_faces(ra.faces[fa], rb.faces[fb], linking_length, kdtree_workers))(chunk_results[idx_of[bid]], chunk_results[idx_of[nb]]))

    all_edges = dask.delayed(_concat_edges)(*edge_tasks)
    mapping_kv = dask.delayed(_build_mapping_from_edges)(all_edges)
    chunk_cats_final = [dask.delayed(_finalize_chunk_catalog)(cr, mapping_kv[0], mapping_kv[1]) for cr in chunk_results]
    catalog = dask.delayed(_merge_catalog_dicts)(chunk_cats_final)
    return catalog

def catalog_to_wcs_table(cat: dict, wcs_header=None) -> Table:
    if "id" in cat and cat["id"].size: order = np.argsort(cat["id"])
    else: order = slice(None)
    tab = Table()
    for key, val in cat.items(): tab[key] = val[order]
    
    # Geometrische Zentren berechnen
    tab["x_center"] = (tab["xmin"] + tab["xmax"]) / 2.0
    tab["y_center"] = (tab["ymin"] + tab["ymax"]) / 2.0
    tab["z_center"] = (tab["zmin"] + tab["zmax"]) / 2.0

    if wcs_header is not None:
        wcs = wcs_header if isinstance(wcs_header, WCS) else WCS(wcs_header)
        x, y, z = np.asarray(tab["x_center"]), np.asarray(tab["y_center"]), np.asarray(tab["z_center"])
        world = wcs.all_pix2world(x, y, z, 0)
        tab["ra"] = world[0]
        tab["dec"] = world[1]
        tab["z"] = world[2]
    return tab
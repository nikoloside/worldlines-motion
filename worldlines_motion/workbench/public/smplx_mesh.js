// SMPL-X in the browser: CPU linear blend skinning of the official mesh served by the workbench.
// The server never ships the model: it sends rest vertices (already shaped/sculpted), faces,
// rest joints and a sparse weight table, all computed locally from the user's licensed NPZ.
import * as THREE from 'three';

const _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _m = new THREE.Matrix4();

export function aaToQuat(x, y, z, out = new THREE.Quaternion()) {
  const a = Math.hypot(x, y, z);
  if (a < 1e-8) return out.set(0, 0, 0, 1);
  return out.setFromAxisAngle(_v.set(x / a, y / a, z / a), a);
}

export class SmplxMesh {
  constructor(payload, material) {
    this.faces = payload.faces;
    this.parents = payload.parents;
    this.influences = payload.influences;
    this.skinIndex = payload.skin_index;
    this.skinWeight = payload.skin_weight;
    this.setRest(payload.vertices, payload.joints);
    this.geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(this.rest.length);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setIndex(this.faces);
    this.mesh = new THREE.Mesh(this.geometry, material || new THREE.MeshStandardMaterial({ color: 0xcdbcae, roughness: 0.72, metalness: 0.02 }));
    this.mesh.castShadow = true; this.mesh.frustumCulled = false;
    this.J = this.rest.length / 3 > 0 ? this.restJoints.length / 3 : 55;
    this.G = Array.from({ length: this.J }, () => new THREE.Matrix4());
    this.A = Array.from({ length: this.J }, () => new THREE.Matrix4());
    this.joints = new Float32Array(this.J * 3);
    this.pose(new Float32Array(3), new Float32Array(63), [0, 0, 0]);
  }

  setRest(vertices, joints) {
    this.rest = Float32Array.from(vertices);
    this.restJoints = Float32Array.from(joints);
    this.dirty = true;
  }

  /** go[3], bp[63] (axis-angle), transl[3]; optional lh/rh[45] and jaw[3]. */
  pose(go, bp, transl, lh = null, rh = null, jaw = null) {
    const J = this.J, R = this.restJoints, P = this.parents;
    for (let j = 0; j < J; j++) {
      let x = 0, y = 0, z = 0;
      if (j === 0) { x = go[0]; y = go[1]; z = go[2]; }
      else if (j < 22) { const k = (j - 1) * 3; x = bp[k]; y = bp[k + 1]; z = bp[k + 2]; }
      else if (j === 22 && jaw) { x = jaw[0]; y = jaw[1]; z = jaw[2]; }
      else if (j >= 25 && j < 40 && lh) { const k = (j - 25) * 3; x = lh[k]; y = lh[k + 1]; z = lh[k + 2]; }
      else if (j >= 40 && rh) { const k = (j - 40) * 3; x = rh[k]; y = rh[k + 1]; z = rh[k + 2]; }
      aaToQuat(x, y, z, _q);
      const p = P[j];
      const rx = R[j * 3] - (p >= 0 ? R[p * 3] : 0), ry = R[j * 3 + 1] - (p >= 0 ? R[p * 3 + 1] : 0), rz = R[j * 3 + 2] - (p >= 0 ? R[p * 3 + 2] : 0);
      _m.compose(_v.set(rx, ry, rz), _q, new THREE.Vector3(1, 1, 1));
      if (p >= 0) this.G[j].multiplyMatrices(this.G[p], _m); else this.G[j].copy(_m);
      const e = this.G[j].elements;
      this.joints[j * 3] = e[12] + transl[0]; this.joints[j * 3 + 1] = e[13] + transl[1]; this.joints[j * 3 + 2] = e[14] + transl[2];
      // A = G with the rest joint removed: A.t = G.t - G.R * restJoint
      const a = this.A[j].copy(this.G[j]).elements;
      const jx = R[j * 3], jy = R[j * 3 + 1], jz = R[j * 3 + 2];
      a[12] -= a[0] * jx + a[4] * jy + a[8] * jz;
      a[13] -= a[1] * jx + a[5] * jy + a[9] * jz;
      a[14] -= a[2] * jx + a[6] * jy + a[10] * jz;
    }
    const V = this.rest.length / 3, K = this.influences, out = this.positions, rest = this.rest;
    const SI = this.skinIndex, SW = this.skinWeight;
    for (let v = 0; v < V; v++) {
      const x = rest[v * 3], y = rest[v * 3 + 1], z = rest[v * 3 + 2];
      let ox = 0, oy = 0, oz = 0;
      for (let k = 0; k < K; k++) {
        const w = SW[v * K + k]; if (w < 1e-6) continue;
        const a = this.A[SI[v * K + k]].elements;
        ox += w * (a[0] * x + a[4] * y + a[8] * z + a[12]);
        oy += w * (a[1] * x + a[5] * y + a[9] * z + a[13]);
        oz += w * (a[2] * x + a[6] * y + a[10] * z + a[14]);
      }
      out[v * 3] = ox + transl[0]; out[v * 3 + 1] = oy + transl[1]; out[v * 3 + 2] = oz + transl[2];
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.geometry.computeBoundingSphere();
  }

  joint(j, out = new THREE.Vector3()) { return out.set(this.joints[j * 3], this.joints[j * 3 + 1], this.joints[j * 3 + 2]); }

  lowestY() { let lo = Infinity; const p = this.positions; for (let i = 1; i < p.length; i += 3) if (p[i] < lo) lo = p[i]; return lo; }
}

/** Simple joint skeleton overlay (lines parent->child) for the 22 body joints. */
export class SkeletonLines {
  constructor(parents, color = 0x5ab8ff) {
    this.parents = parents; this.count = 22;
    const pos = new Float32Array(this.count * 2 * 3);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.lines = new THREE.LineSegments(this.geometry, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 }));
    this.lines.renderOrder = 10; this.lines.frustumCulled = false;
  }
  update(joints) {
    const p = this.geometry.attributes.position.array;
    for (let j = 1; j < this.count; j++) {
      const par = this.parents[j], o = j * 6;
      p[o] = joints[par * 3]; p[o + 1] = joints[par * 3 + 1]; p[o + 2] = joints[par * 3 + 2];
      p[o + 3] = joints[j * 3]; p[o + 4] = joints[j * 3 + 1]; p[o + 5] = joints[j * 3 + 2];
    }
    this.geometry.attributes.position.needsUpdate = true;
  }
}

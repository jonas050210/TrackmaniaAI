/**
 * Interactive 3D track viewer (three.js).
 *
 * Renders the track centreline, the corridor ribbon (left/right edges), curvature colouring,
 * optional trajectories (replays / ghosts) and a car marker that follows a sample index for
 * playback. Purely presentational: all data comes in as props.
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { TrackGeometry } from "../api";

export interface Trajectory {
  positions: number[][];
  label: string;
  color: number;
}

export interface Props {
  geometry: TrackGeometry | null;
  trajectories?: Trajectory[];
  /** Sample index to show the car at, when a trajectory is present. */
  carIndex?: number;
  height?: number;
  showCorridor?: boolean;
  showCurvature?: boolean;
  onCarIndexChange?: (index: number) => void;
}

function toVector3Array(points: number[][]): THREE.Vector3[] {
  return points.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
}

export function TrackViewer3D({
  geometry,
  trajectories = [],
  carIndex,
  height = 520,
  showCorridor = true,
  showCurvature = true,
  onCarIndexChange,
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const carsRef = useRef<THREE.Mesh[]>([]);
  const [ready, setReady] = useState(false);

  // Build the whole scene when the inputs change.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x080b10);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x080b10, 400, 1600);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(120, 220, 80);
    scene.add(sun);

    const grid = new THREE.GridHelper(1200, 60, 0x1c2634, 0x131c28);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    scene.add(grid);

    const group = new THREE.Group();
    scene.add(group);

    if (geometry) {
      const points = toVector3Array(geometry.points);

      // corridor ribbon
      if (showCorridor && geometry.edges.left.length === points.length) {
        const left = toVector3Array(geometry.edges.left);
        const right = toVector3Array(geometry.edges.right);
        const positions = new Float32Array(points.length * 2 * 3);
        const indices: number[] = [];
        for (let i = 0; i < points.length; i++) {
          positions.set([left[i].x, left[i].y + 0.02, left[i].z], i * 6);
          positions.set([right[i].x, right[i].y + 0.02, right[i].z], i * 6 + 3);
          if (i < points.length - 1 || geometry.closed) {
            const next = (i + 1) % points.length;
            const a = i * 2;
            const b = next * 2;
            indices.push(a, a + 1, b, a + 1, b + 1, b);
          }
        }
        const meshGeometry = new THREE.BufferGeometry();
        meshGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        meshGeometry.setIndex(indices);
        meshGeometry.computeVertexNormals();
        group.add(
          new THREE.Mesh(
            meshGeometry,
            new THREE.MeshStandardMaterial({
              color: 0x2a3a4d,
              roughness: 0.9,
              metalness: 0.05,
              side: THREE.DoubleSide,
            })
          )
        );
      }

      // centreline, coloured by curvature when available
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const hasCurvature = showCurvature && geometry.curvature.length === points.length;
      if (hasCurvature) {
        const maxCurvature = Math.max(...geometry.curvature.map(Math.abs), 1e-6);
        const colors = new Float32Array(points.length * 3);
        const straight = new THREE.Color(0x38bdf8);
        const corner = new THREE.Color(0xfbbf24);
        for (let i = 0; i < points.length; i++) {
          const t = Math.min(1, Math.abs(geometry.curvature[i]) / maxCurvature);
          const color = straight.clone().lerp(corner, t);
          colors.set([color.r, color.g, color.b], i * 3);
        }
        lineGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      }
      const centerlineMaterial = new THREE.LineBasicMaterial({
        color: hasCurvature ? 0xffffff : 0x38bdf8,
        vertexColors: hasCurvature,
      });
      group.add(
        geometry.closed
          ? new THREE.LineLoop(lineGeometry, centerlineMaterial)
          : new THREE.Line(lineGeometry, centerlineMaterial)
      );

      // start marker
      const start = new THREE.Mesh(
        new THREE.CylinderGeometry(0.4, 0.4, 6, 16),
        new THREE.MeshStandardMaterial({
          color: 0x34d399,
          emissive: 0x34d399,
          emissiveIntensity: 0.4,
        })
      );
      start.position.copy(points[0]);
      start.position.y += 3;
      group.add(start);

      // frame the track
      const box = new THREE.Box3().setFromPoints(points);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) / 2 || 100;
      camera.position.set(center.x + radius * 1.1, radius * 0.9, center.z + radius * 1.1);
      controls.target.copy(center);
      controls.update();
    }

    // trajectories + car markers. If an old replay has no matching track geometry, frame the
    // recorded path itself rather than silently substituting a synthetic map.
    const cars: THREE.Mesh[] = [];
    const trajectoryBounds = new THREE.Box3();
    let hasTrajectoryPoints = false;
    trajectories.forEach((trajectory) => {
      const points = toVector3Array(trajectory.positions);
      if (points.length < 2) return;
      if (!geometry) {
        for (const point of points) trajectoryBounds.expandByPoint(point);
        hasTrajectoryPoints = true;
      }
      group.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineBasicMaterial({ color: trajectory.color })
        )
      );
      const car = new THREE.Mesh(
        new THREE.BoxGeometry(2.2, 0.9, 4.4),
        new THREE.MeshStandardMaterial({
          color: trajectory.color,
          emissive: trajectory.color,
          emissiveIntensity: 0.35,
        })
      );
      car.visible = false;
      group.add(car);
      cars.push(car);
    });
    if (!geometry && hasTrajectoryPoints) {
      const center = trajectoryBounds.getCenter(new THREE.Vector3());
      const size = trajectoryBounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z) / 2 || 100;
      camera.position.set(center.x + radius * 1.1, center.y + radius * 0.9, center.z + radius * 1.1);
      controls.target.copy(center);
      controls.update();
    }
    carsRef.current = cars;

    const resize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    setReady(true);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        const renderable = object as THREE.Object3D & {
          geometry?: THREE.BufferGeometry;
          material?: THREE.Material | THREE.Material[];
        };
        renderable.geometry?.dispose();
        if (Array.isArray(renderable.material)) {
          renderable.material.forEach((material) => material.dispose());
        } else {
          renderable.material?.dispose();
        }
      });
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      carsRef.current = [];
      setReady(false);
    };
  }, [geometry, trajectories, showCorridor, showCurvature]);

  // position the car markers for the requested sample index
  useEffect(() => {
    if (carIndex === undefined) return;
    const cars = carsRef.current;
    trajectories.forEach((trajectory, i) => {
      const car = cars[i];
      if (!car) return;
      const index = Math.min(carIndex, trajectory.positions.length - 1);
      if (index < 0) {
        car.visible = false;
        return;
      }
      const p = trajectory.positions[index];
      const next = trajectory.positions[Math.min(index + 1, trajectory.positions.length - 1)];
      car.position.set(p[0], p[1] + 0.6, p[2]);
      car.lookAt(next[0], next[1] + 0.6, next[2]);
      car.visible = true;
    });
  }, [carIndex, trajectories]);

  const maxIndex = trajectories.reduce((max, t) => Math.max(max, t.positions.length - 1), 0);

  return (
    <div className="track-viewer" style={{ height }}>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />
      {geometry && (
        <div className="overlay">
          <strong>{geometry.name}</strong> · {geometry.length.toFixed(0)} m · {geometry.num_points} samples
          {showCurvature && geometry.curvature.length > 0 && (
            <span>
              {" "}· <span style={{ color: "#38bdf8" }}>■</span> straight{" "}
              <span style={{ color: "#fbbf24" }}>■</span> corners
            </span>
          )}
        </div>
      )}
      {!geometry && (
        <div className="overlay">
          {trajectories.length > 0
            ? "Track geometry unavailable · replay trajectory only"
            : "No track or replay selected"}
        </div>
      )}
      {!ready && <div className="overlay">initialising 3D…</div>}
      {carIndex !== undefined && onCarIndexChange && maxIndex > 0 && (
        <div className="controls">
          <button className="btn sm" onClick={() => onCarIndexChange(Math.max(0, carIndex - 1))}>
            ◀
          </button>
          <input
            type="range"
            min={0}
            max={maxIndex}
            value={carIndex}
            onChange={(event) => onCarIndexChange(Number(event.target.value))}
          />
          <button
            className="btn sm"
            onClick={() => onCarIndexChange(Math.min(maxIndex, carIndex + 1))}
          >
            ▶
          </button>
          <span className="time">
            {carIndex} / {maxIndex}
          </span>
        </div>
      )}
    </div>
  );
}

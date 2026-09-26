import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Focus, Move3D, Maximize, Minimize } from "lucide-react";
import type { Playback, RobotModel } from "./shared";

export default function RobotViewer({
  model,
  playback,
}: {
  model: RobotModel | null;
  playback: Playback | null;
}) {
  const mount = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [fullscreen, setFullscreen] = useState(false);
  useEffect(() => {
    const change = () => setFullscreen(document.fullscreenElement === stage.current);
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !document.fullscreenElement) setFullscreen(false);
    };
    document.addEventListener("fullscreenchange", change);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("fullscreenchange", change);
      document.removeEventListener("keydown", escape);
    };
  }, []);
  async function toggleFullscreen() {
    if (fullscreen) {
      if (document.fullscreenElement === stage.current) await document.exitFullscreen();
      setFullscreen(false);
    } else {
      setFullscreen(true);
      try { await stage.current?.requestFullscreen?.(); }
      catch { /* Keep the expanded viewport when native fullscreen is unavailable. */ }
    }
  }
  const pose = useRef(playback);
  const recenter = useRef<() => void>(() => {});
  const [error, setError] = useState("");
  pose.current = playback;
  useEffect(() => {
    if (!mount.current || !model) return;
    setError("");
    const host = mount.current;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setError(
        "WebGL is unavailable. Enable hardware acceleration to see the robot.",
      );
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x191e1c);
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.up.set(0, 0, 1);
    scene.fog = new THREE.Fog(0x191e1c, 9, 24);
    const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(1.65, -2.1, 1.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.65);
    controls.enableDamping = true;
    controls.minDistance = 0.6;
    controls.maxDistance = 12;
    controls.maxPolarAngle = Math.PI * 0.49;
    const ambient = new THREE.HemisphereLight(0xf1fff6, 0x353f38, 2.4);
    scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 3.5);
    sun.position.set(3, -4, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -6;
    sun.shadow.camera.right = 6;
    sun.shadow.camera.top = 6;
    sun.shadow.camera.bottom = -6;
    sun.shadow.normalBias = 0.015;
    scene.add(sun, sun.target);
    const rim = new THREE.DirectionalLight(0xc4f28c, 1.8);
    rim.position.set(-2, 3, 3);
    scene.add(rim);
    const floorGeometry = new THREE.PlaneGeometry(100, 100);
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0x202622,
      roughness: 0.95,
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.position.z = -0.009;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(40, 80, 0x56604f, 0x333d34);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.005;
    scene.add(grid);
    const geometries = model.meshes.map((m) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(m.vertices, 3),
      );
      geometry.setIndex(m.indices);
      geometry.computeVertexNormals();
      return geometry;
    });
    const bodies = new Map<number, THREE.Group>();
    const materials: THREE.Material[] = [];
    for (const geom of model.geoms) {
      let body = bodies.get(geom.body);
      if (!body) {
        body = new THREE.Group();
        bodies.set(geom.body, body);
        scene.add(body);
      }
      let geometry: THREE.BufferGeometry | undefined;
      if (geom.type === 7) geometry = geometries[geom.mesh];
      else if (geom.type === 2)
        geometry = new THREE.SphereGeometry(geom.size[0], 16, 12);
      else if (geom.type === 6)
        geometry = new THREE.BoxGeometry(
          ...(geom.size.map((v) => v * 2) as [number, number, number]),
        );
      else if (geom.type === 3 || geom.type === 5) {
        geometry =
          geom.type === 3
            ? new THREE.CapsuleGeometry(geom.size[0], geom.size[1] * 2, 6, 12)
            : new THREE.CylinderGeometry(
                geom.size[0],
                geom.size[0],
                geom.size[1] * 2,
                16,
              );
        geometry.rotateX(Math.PI / 2);
      }
      if (!geometry) continue;
      if (geom.type !== 7) geometries.push(geometry);
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(geom.color[0], geom.color[1], geom.color[2]),
        roughness: 0.55,
        metalness: 0.2,
      });
      materials.push(material);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.fromArray(geom.position);
      mesh.quaternion.set(
        geom.quaternion[1],
        geom.quaternion[2],
        geom.quaternion[3],
        geom.quaternion[0],
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      body.add(mesh);
    }
    recenter.current = () => {
      const p = pose.current?.positions;
      const root = model.root_body || 1;
      const x = p?.[root * 3] || 0,
        y = p?.[root * 3 + 1] || 0;
      controls.target.set(x, y, 0.65);
      camera.position.set(x + 1.65, y - 2.1, 1.4);
      controls.update();
    };
    const resize = new ResizeObserver(() => {
      const { width, height } = host.getBoundingClientRect();
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    });
    resize.observe(host);
    let frame: number;
    const target = new THREE.Vector3(),
      quaternion = new THREE.Quaternion();
    const previousRoot = new THREE.Vector3(),
      delta = new THREE.Vector3();
    const draw = () => {
      const current = pose.current;
      const matching =
        current && (current.model_id || "g1") === (model.id || "g1");
      for (const body of bodies.values()) body.visible = !!matching;
      if (matching && current)
        for (const [id, body] of bodies) {
          target.fromArray(current.positions, id * 3);
          const q = current.quaternions,
            offset = id * 4;
          quaternion.set(
            q[offset + 1],
            q[offset + 2],
            q[offset + 3],
            q[offset],
          );
          body.position.lerp(target, 0.6);
          body.quaternion.slerp(quaternion, 0.6);
        }
      const pelvis = bodies.get(model.root_body || 1);
      if (pelvis) {
        delta.set(
          pelvis.position.x - previousRoot.x,
          pelvis.position.y - previousRoot.y,
          0,
        );
        camera.position.add(delta);
        controls.target.add(delta);
        sun.position.add(delta);
        sun.target.position.add(delta);
        previousRoot.copy(pelvis.position);
      }
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      floorGeometry.dispose();
      floorMaterial.dispose();
      grid.geometry.dispose();
      if (Array.isArray(grid.material))
        grid.material.forEach((m) => m.dispose());
      else grid.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [model]);
  return (
    <div ref={stage} className={`viewer-stage ${fullscreen ? "viewer-fullscreen" : ""}`}>
      <div
        ref={mount}
        className="viewer-canvas"
        aria-label="Interactive 3D Unitree G1 robot. Drag to orbit, scroll to zoom."
      />
      {(!model ||
        (model.id || "g1") !== (playback?.model_id || "g1") ||
        error) && (
        <div className="viewer-loading">
          <Move3D size={32} />
          <p>{error || "Loading the G1 model…"}</p>
        </div>
      )}
      <div className="viewport-label">
        <span className="live-dot" /> LIVE VIEW <span className="muted">/</span>{" "}
        UNITREE G1
      </div>
      <div className="viewport-mode">
        {playback?.mode === "wbc_tracking"
          ? "PHYSICS · WBC TRACKING"
          : playback?.mode === "motor_driven_physics"
            ? "PHYSICS TRIAL"
            : "MOTION PREVIEW"}
      </div>
      <div className="viewport-bottom">
        <span>
          29 DOF <i /> Z UP <i /> METRES
        </span>
        <button
          className="icon-button"
          onClick={() => void toggleFullscreen()}
          aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          title={fullscreen ? "Exit fullscreen (Esc)" : "Enter fullscreen"}
        >
          {fullscreen ? <Minimize size={17} /> : <Maximize size={17} />}
        </button>
        <button
          className="icon-button"
          onClick={() => recenter.current()}
          aria-label="Recenter camera"
          title="Recenter camera"
        >
          <Focus size={17} />
        </button>
      </div>
      <div className="orbit-hint">Drag to orbit · Scroll to zoom</div>
    </div>
  );
}

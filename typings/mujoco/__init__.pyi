from typing import Any, Sequence
import numpy as np

class MjOption:
    timestep: float
    gravity: np.ndarray
    def __getattr__(self, name: str) -> Any: ...

class MjModel:
    nv: int
    nq: int
    nu: int
    nbody: int
    opt: MjOption
    actuator_ctrlrange: np.ndarray | None
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def from_xml_path(cls, filename: str) -> MjModel: ...
    @classmethod
    def from_xml_string(cls, xml: str) -> MjModel: ...
    def __getattr__(self, name: str) -> Any: ...

class MjData:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    ctrl: np.ndarray
    qfrc_applied: np.ndarray
    xpos: np.ndarray
    def __init__(self, model: MjModel) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class mjtObj:
    mjOBJ_UNKNOWN: int
    mjOBJ_BODY: int
    mjOBJ_XBODY: int
    mjOBJ_JOINT: int
    mjOBJ_DOF: int
    mjOBJ_GEOM: int
    mjOBJ_SITE: int
    mjOBJ_CAMERA: int
    mjOBJ_LIGHT: int
    mjOBJ_MESH: int
    mjOBJ_SKIN: int
    mjOBJ_HFIELD: int
    mjOBJ_TEXTURE: int
    mjOBJ_MATERIAL: int
    mjOBJ_PAIR: int
    mjOBJ_EXCLUDE: int
    mjOBJ_EQUALITY: int
    mjOBJ_TENDON: int
    mjOBJ_ACTUATOR: int
    mjOBJ_SENSOR: int
    mjOBJ_NUMERIC: int
    mjOBJ_TEXT: int
    mjOBJ_TUPLE: int
    mjOBJ_KEY: int
    def __getattr__(self, name: str) -> Any: ...

def mj_step(m: MjModel, d: MjData) -> None: ...
def mj_step1(m: MjModel, d: MjData) -> None: ...
def mj_step2(m: MjModel, d: MjData) -> None: ...
def mj_forward(m: MjModel, d: MjData) -> None: ...
def mj_resetData(m: MjModel, d: MjData) -> None: ...
def mj_name2id(m: MjModel, type: int, name: str) -> int: ...
def mj_id2name(m: MjModel, type: int, id: int) -> str: ...
def mj_jacBody(m: MjModel, d: MjData, jacp: np.ndarray | None, jacr: np.ndarray | None, body: int) -> None: ...
def mj_jacSite(m: MjModel, d: MjData, jacp: np.ndarray | None, jacr: np.ndarray | None, site: int) -> None: ...

viewer: Any

def __getattr__(name: str) -> Any: ...

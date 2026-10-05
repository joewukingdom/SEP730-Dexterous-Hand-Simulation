"""Generate the Panda asset file and the ORCA-on-Panda scenes from mujoco_menagerie.

Usage:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
    git -C mujoco_menagerie sparse-checkout set franka_emika_panda
    python scripts/gen_arm_scenes.py mujoco_menagerie/franka_emika_panda .
"""
import copy, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path

src = Path(sys.argv[1]); repo = Path(sys.argv[2])
model_dir = repo / "src/orca_sim/models/franka_panda"
scene_dir = repo / "src/orca_sim/scenes/v2"
root = ET.parse(src / "panda_nohand.xml").getroot()
A = "panda_"  # prefix for shared assets (materials / meshes)

# ---------- shared assets: defaults + materials + meshes ----------
defaults = root.find("default")
panda_cls = defaults.find("default")
panda_cls.remove(panda_cls.find("default[@class='finger']"))  # gripper only
ET.SubElement(panda_cls, "mesh", scale="1 1 1")  # override ORCA's global mesh scale=0.001
asset = root.find("asset")
used = []
for el in asset:
    if el.tag == "material":
        el.set("name", A + el.get("name"))
    else:
        f = el.get("file")
        el.set("name", A + el.get("name", Path(f).stem))
        el.set("file", "assets/" + f)
        el.set("class", "panda")
        used.append(f)
(model_dir / "assets").mkdir(parents=True, exist_ok=True)
for f in used:
    shutil.copy(src / "assets" / f, model_dir / "assets" / f)
shutil.copy(src / "LICENSE", model_dir / "LICENSE")
out = ET.Element("mujoco", model="franka_panda_assets")
out.append(defaults); out.append(asset)
ET.indent(out, "  ")
(model_dir / "panda.mjcf").write_text(
    "<?xml version='1.0' encoding='utf-8'?>\n"
    "<!-- Franka Panda (no gripper) shared assets, adapted from mujoco_menagerie/franka_emika_panda/\n"
    "     panda_nohand.xml (Apache-2.0, see LICENSE). Materials/meshes prefixed with 'panda_' and\n"
    "     meshes forced to scale 1 so they can be combined with the ORCA hand models.\n"
    "     Body trees and actuators are instantiated per arm in the scene files. -->\n"
    + ET.tostring(out, encoding="unicode") + "\n", encoding="utf-8")

# ---------- per-arm instance: body tree, actuators, contacts ----------
def arm(side: str, base_pos: str, mount_extras: str = ""):
    P = f"{side}_panda_"
    body = copy.deepcopy(root.find("worldbody").find("body"))
    act = copy.deepcopy(root.find("actuator"))
    con = copy.deepcopy(root.find("contact"))
    for tree in (body, act, con):
        for el in tree.iter():
            for attr in ("joint", "body1", "body2"):
                if attr in el.attrib:
                    el.set(attr, P + el.get(attr))
            if el.tag in ("body", "joint", "general", "site") and "name" in el.attrib:
                el.set("name", P + el.get("name"))
            if "material" in el.attrib:
                el.set("material", A + el.get("material"))
            if "mesh" in el.attrib:
                el.set("mesh", A + el.get("mesh"))
    body.set("pos", base_pos)
    attach = body.find(f".//body[@name='{P}attachment']")
    attach.append(ET.Comment(" ORCA forearm base sits flush on the Panda flange; childclass=\"main\""
                             " stops the hand from inheriting the Panda defaults "))
    mount = ET.SubElement(attach, "body", name=f"{side}_mount", pos="0.01 0 0.058",
                          euler="1.5708 0 0", childclass="main")
    ET.SubElement(mount, "include", file=f"../../models/v2/mjcf/orcahand_{side}_body.xml")
    # Extra sites / cameras rigidly attached to the hand (mount frame == right_tower frame)
    mount.extend(ET.fromstring(f"<x>{mount_extras.format(side=side)}</x>"))
    con.append(ET.Element("exclude", body1=f"{P}link7", body2=f"{side}_ForeArmStructure-Model_e18f2368"))
    con.append(ET.Element("exclude", body1=f"{P}link6", body2=f"{side}_ForeArmStructure-Model_e18f2368"))
    return body, list(act), list(con)

ARM_HOME = "0 0 0 -1.57079 0 1.57079 -0.7853"
HAND_ZERO = " ".join(["0"] * 17)

def scene(name, title, arms, *, mount_extras="", world_extras="", top_extras="",
          extra_qpos="", center="0.3 0 0.4"):
    """arms: [(side, base_pos)]. world_extras: XML appended to <worldbody> after the arms
    (free bodies there must list their initial qpos in extra_qpos, in order)."""
    m = ET.Element("mujoco", model=title)
    for f in ["../../models/v2/assets/scene.xml", "../../models/v2/assets/options.xml",
              "../../models/franka_panda/panda.mjcf"]:
        ET.SubElement(m, "include", file=f)
    acts = ET.Element("actuator"); cons = ET.Element("contact")
    bodies = []
    for side, pos in arms:
        body, a, c = arm(side, pos, mount_extras)
        bodies.append(body); acts.extend(a); cons.extend(c)
    # Actuators are numbered in parse order: declare the Panda ones before
    # including the hand files so ctrl = [arms..., hands...]
    m.append(ET.Comment(" Panda actuators come before the hand includes so ctrl = [arms..., hands...] "))
    m.append(acts)
    for s, _ in arms:
        ET.SubElement(m, "include", file=f"../../models/v2/mjcf/orcahand_{s}.mjcf")
    m.append(ET.Comment(" Stiff Panda PD gains need an implicit integrator "))
    ET.SubElement(m, "option", integrator="implicitfast", impratio="10")
    ET.SubElement(m, "statistic", extent="1.2", center=center)
    m.extend(ET.fromstring(f"<x>{top_extras}</x>"))
    wb = ET.SubElement(m, "worldbody")
    wb.extend(bodies)
    wb.extend(ET.fromstring(f"<x>{world_extras}</x>"))
    m.append(cons)
    order = " + ".join(f"{s} Panda (7)" for s, _ in arms) + " + " + \
            " + ".join(f"{s} hand (17)" for s, _ in arms)
    kf = ET.SubElement(m, "keyframe")
    kf.append(ET.Comment(f" ctrl order: {order}. qpos follows the body tree: each Panda then its hand. "))
    qpos = " ".join([f"{ARM_HOME} {HAND_ZERO}" for _ in arms] + ([extra_qpos] if extra_qpos else []))
    ctrl = " ".join([ARM_HOME] * len(arms) + [HAND_ZERO] * len(arms))
    ET.SubElement(kf, "key", name="home", qpos=qpos, ctrl=ctrl)
    ET.indent(m, "  ")
    (scene_dir / name).write_text('<?xml version="1.0" ?>\n' + ET.tostring(m, encoding="unicode") + "\n",
                                  encoding="utf-8")

scene("scene_right_arm.xml", "ORCA hand right on Franka Panda scene", [("right", "0 0 0")])
scene("scene_combined_arm.xml", "ORCA hands on two Franka Pandas scene",
      [("right", "0 -0.3 0"), ("left", "0 0.3 0")])

# ---------- tabletop pick-and-place scene ----------
import math

TABLE_TOP = 0.40   # Panda base sits on the table at this height
BLOCK_HALF = 0.02  # 4 cm cubes


def block(name, rgba, x, y):
    return (f'<body name="{name}" pos="{x} {y} {TABLE_TOP + BLOCK_HALF}">'
            f'<freejoint name="{name}_joint"/>'
            f'<geom name="{name}_geom" type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}"'
            f' rgba="{rgba}" mass="0.05" friction="1.5 0.01 0.002" condim="4"/></body>')


def bowl(name, x, y, r=0.065, wall_h=0.025, n=16):
    walls = []
    seg = 2 * r * math.tan(math.pi / n) / 2 + 0.002
    for i in range(n):
        a = 2 * math.pi * i / n
        walls.append(f'<geom type="box" size="0.003 {seg:.4f} {wall_h / 2}" '
                     f'pos="{r * math.cos(a):.4f} {r * math.sin(a):.4f} {wall_h / 2 + 0.008}" '
                     f'euler="0 0 {a:.4f}" material="bowl"/>')
    return (f'<body name="{name}" pos="{x} {y} {TABLE_TOP}">'
            f'<freejoint name="{name}_joint"/>'
            f'<inertial pos="0 0 0.01" mass="0.3" diaginertia="0.0005 0.0005 0.0008"/>'
            f'<geom type="cylinder" size="{r + 0.003} 0.004" pos="0 0 0.004" material="bowl"/>'
            + "".join(walls) + "</body>")


OBJECTS = [  # (xml, initial qpos "x y z qw qx qy qz") — positions are re-sampled by the env
    (block("red_block", "0.85 0.1 0.1 1", 0.55, -0.12), f"0.55 -0.12 {TABLE_TOP + BLOCK_HALF} 1 0 0 0"),
    (block("green_block", "0.15 0.7 0.2 1", 0.62, 0.0), f"0.62 0.0 {TABLE_TOP + BLOCK_HALF} 1 0 0 0"),
    (block("blue_block", "0.15 0.3 0.85 1", 0.50, 0.08), f"0.50 0.08 {TABLE_TOP + BLOCK_HALF} 1 0 0 0"),
    (bowl("bowl", 0.58, 0.20), f"0.58 0.20 {TABLE_TOP} 1 0 0 0"),
]

TABLETOP_WORLD = (
    f'<geom name="table" type="box" pos="0.45 0 {TABLE_TOP - 0.02}" size="0.55 0.6 0.02" material="table"/>'
    + "".join(f'<geom type="box" pos="{x} {y} {(TABLE_TOP - 0.04) / 2}" size="0.03 0.03 {(TABLE_TOP - 0.04) / 2}" material="table_leg"/>'
              for x in (-0.05, 0.95) for y in (-0.55, 0.55))
    # Fixed third-person camera facing the robot across the table
    + '<camera name="front" pos="1.3 0 0.95" xyaxes="0 1 0 -0.346 0 0.938" fovy="55"/>'
    # Side view (an overhead camera would be blocked by the arm)
    + '<camera name="side" pos="0.5 -1.1 0.85" xyaxes="1 0 0 0 0.303 0.953" fovy="55"/>'
    + "".join(xml for xml, _ in OBJECTS)
)

TABLETOP_TOP = (
    '<visual><global offwidth="1280" offheight="960"/></visual>'
    '<asset>'
    '<texture name="wood" type="2d" builtin="flat" rgb1="0.55 0.42 0.30" width="64" height="64"/>'
    '<material name="table" texture="wood" rgba="1 1 1 1" specular="0.1"/>'
    '<material name="table_leg" rgba="0.3 0.3 0.3 1"/>'
    '<material name="bowl" rgba="0.92 0.92 0.88 1" specular="0.3"/>'
    '</asset>'
)

# Hand-attached frames (in the right_tower frame: fingers along +y, palm faces -z)
HAND_EXTRAS = (
    '<site name="{side}_grasp_site" pos="-0.01 0.17 -0.035" size="0.008" rgba="1 0.5 0 0.6" group="4"/>'
    '<camera name="{side}_wrist" pos="0 -0.04 -0.11" xyaxes="1 0 0 0 0.4 1" fovy="60"/>'
)

scene("scene_right_arm_tabletop.xml", "ORCA right hand on Franka Panda tabletop pick-and-place scene",
      [("right", f"0 0 {TABLE_TOP}")],
      mount_extras=HAND_EXTRAS, world_extras=TABLETOP_WORLD, top_extras=TABLETOP_TOP,
      extra_qpos=" ".join(q for _, q in OBJECTS), center=f"0.5 0 {TABLE_TOP + 0.2}")

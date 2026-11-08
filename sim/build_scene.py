"""Builds the NUCK-3 MuJoCo scene: two MakerMods Metal arms on a table, the Hill Climb props, three cameras.

    repos/dimos/.venv/bin/python sim/build_scene.py        # writes sim/assets/scene.xml

The arm comes from the vendor URDF (makermods-robotics/metal-python-ros, identical to the station's copy). The
layout follows the station photos: both bases on the table edge facing +x, 0.6 m apart (DimOS's default bench
layout, y = +-0.3 m), the power strip and the blocks 0.3-0.5 m in front. Positions are estimates, not measurements:
change MOUNT_Y / the prop ranges in metal_sim.py when the real rig is measured.
"""

import pathlib
import re

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
DESC = REPO / "repos/metal-python-ros/metal_ros2/src/metal_description"
ASSETS = ROOT / "assets"

PORT_Z = 0.010  # the USB socket centre (and the plug housing half-height) above the table, m
MOUNT_Y = 0.30  # each base's distance from the centre line, m
# Tower pieces (m), measured at the station; metal_sim.py imports these.
CUBE_HALF_XY = 0.0254  # 2 in cube
CUBE_HALF_Z = 0.0165  # 1.3 in tall
HOLE_HALF = 0.0112  # the through-hole: 0.88 in square, for a 0.8 in rod
ROD_R = 0.0102  # 0.8 in across
ROD_HALF_LEN = 0.0508  # 4 in long
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def arm_mjcf() -> pathlib.Path:
    """Converts the vendor URDF to a standalone MJCF arm (meshes resolved, finger mimic as an equality)."""
    urdf = (DESC / "urdf/metal_with_gripper.urdf").read_text()
    urdf = urdf.replace("package://metal_description/meshes/", "")
    urdf = re.sub(
        r"<robot([^>]*)>",
        rf'<robot\1><mujoco><compiler meshdir="{DESC / "meshes"}" balanceinertia="true" discardvisual="false" '
        r'fusestatic="false" strippath="false"/></mujoco>',
        urdf,
        count=1,
    )
    tmp = ASSETS / "_metal.urdf"
    tmp.write_text(urdf)
    model = mujoco.MjModel.from_xml_path(str(tmp))
    out = ASSETS / "metal_arm_raw.xml"
    mujoco.mj_saveLastXML(str(out), model)
    tmp.unlink()
    return out


def build() -> pathlib.Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    raw = arm_mjcf()

    spec = mujoco.MjSpec()
    spec.modelname = "nuck3_hill_climb"
    spec.meshdir = str(DESC / "meshes")
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    tex = spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.86, 0.84, 0.78],
                           width=64, height=64)
    mat = spec.add_material(name="table")
    mat.rgba = [0.88, 0.86, 0.8, 1]
    del tex

    world = spec.worldbody
    world.add_light(pos=[0.3, 0, 1.6], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(pos=[1.2, 0.8, 1.2], dir=[-0.6, -0.4, -0.7], diffuse=[0.3, 0.3, 0.3])
    # Table: top surface at z = 0, from just behind the bases to 0.9 m in front.
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.75, 0.02],
                   pos=[0.40, 0, -0.02], material="table", friction=[0.8, 0.01, 0.001], conaffinity=3)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                   rgba=[0.35, 0.35, 0.38, 1])

    for side, y in (("left", MOUNT_Y), ("right", -MOUNT_Y)):
        arm = mujoco.MjSpec.from_file(str(raw))
        # Position-servo every arm joint (the real driver runs MIT position control at kp 100 / kd 5 on the big
        # joints); the fingers get one servo on joint7 and joint8 follows through an equality.
        for j in arm.joints:
            j.armature = 0.02
            j.damping = [1.0 if j.name in ("joint7", "joint8") else 2.0, 0, 0]
            j.frictionloss = 0.05
        for name, kp in (("joint1", 400), ("joint2", 600), ("joint3", 500), ("joint4", 250), ("joint5", 120),
                         ("joint6", 120)):
            jnt = arm.joint(name)
            act = arm.add_actuator(name=f"act_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=kp, kv=0.08 * kp)
            act.ctrlrange = jnt.range
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-120, 120]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        grip = arm.add_actuator(name="act_gripper", target="joint7", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        grip.set_to_position(kp=800, kv=40)
        grip.ctrlrange = [-0.05, 0.0]
        grip.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        grip.forcerange = [-40, 40]
        grip.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        arm.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="joint8", name2="joint7",
                         data=[0, -1, 0, 0, 0] + [0] * 6)
        # Collision groups: arm geoms are bit 2 and only collide with bit-1 things (table, props), never with each
        # other (the vendor mesh hulls overlap between non-adjacent links and would jam the joints). The base and
        # shoulder never collide at all (the base sits on the table). Grippy fingertips.
        for g in arm.geoms:
            g.friction = [1.2, 0.02, 0.002]
            g.condim = 4 if g.parent.name in ("link7", "link8") else 3
            if g.contype or g.conaffinity:
                quiet = g.parent.name in ("base_link", "link1")
                g.contype, g.conaffinity = (0, 0) if quiet else (2, 1)
        link6 = arm.body("link6")
        # Dream's tool point for the stock gripper: [0.135, 0, 0] in link6 (the kit's templates).
        link6.add_site(name="tcp", pos=[0.135, 0, 0], size=[0.006, 0, 0], rgba=[1, 0, 0, 0.6])
        # Wrist camera: on the gripper housing, looking along the fingers (estimated mount).
        cam = link6.add_camera(name="wrist", pos=[0.05, 0, 0.06], fovy=78)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [0, -1, 0, 0.55, 0, 0.83]

        frame = world.add_frame(pos=[0, y, 0])
        frame.attach_body(arm.body("base_link"), f"{side}_", "")
        # attach_body copies only the subtree; actuators/equalities come with it in MuJoCo >= 3.2

    # Fixed overhead ("scene") camera: behind and above the bases, looking forward over the workspace, so both
    # grippers sit at the bottom corners of the frame (as on the station's usb-0:2 camera).
    world.add_camera(name="overhead", pos=[-0.32, 0.0, 0.78], fovy=66,
                     xyaxes=_look_at_xyaxes([-0.32, 0.0, 0.78], [0.42, 0.0, 0.0]))

    _add_props(world, spec)
    for g in spec.geoms:
        if g.parent.name.startswith(("cube_", "rod_", "strip", "usb_plug", "ball_", "basket")):
            g.conaffinity = 3
    out = ASSETS / "scene.xml"
    spec.compile()
    out.write_text(spec.to_xml())
    return out


def _look_at_xyaxes(pos: list[float], target: list[float]) -> list[float]:
    """MuJoCo camera xyaxes (image right, image up) for a camera at pos looking at target, world z up."""
    import numpy as np

    forward = np.subtract(target, pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def _add_props(world: mujoco.MjsBody, spec: mujoco.MjSpec) -> None:
    """Foam cubes with a centre hole, foam rods, and the power strip with a USB plug (free bodies)."""
    cube_colors = {"blue": [0.25, 0.55, 0.85, 1], "green": [0.55, 0.8, 0.3, 1], "purple": [0.78, 0.45, 0.85, 1]}
    # Measured at the station (harness/calibration.yaml): cubes 2 x 2 x 1.3 in lying with a 2 x 2 face up, a hole
    # straight through them; rods 0.8 in across x 4 in. The real tower is three cubes, then a rod standing in the top
    # cube's hole (resting on the cube below): 3 x 1.3 + (4 - 1.3) = 6.6 in.
    s, sz, hole = CUBE_HALF_XY, CUBE_HALF_Z, HOLE_HALF
    wall = (s - hole) / 2
    for i, (name, rgba) in enumerate(cube_colors.items()):
        b = world.add_body(name=f"cube_{name}", pos=[0.45, 0.1 * (i - 1), sz])
        b.add_freejoint(name=f"cube_{name}_free")
        # Four walls around a vertical square through-hole.
        for k, (dx, dy, hx, hy) in enumerate(((s - wall, 0, wall, s), (-(s - wall), 0, wall, s),
                                              (0, s - wall, hole, wall), (0, -(s - wall), hole, wall))):
            b.add_geom(name=f"cube_{name}_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, sz],
                       pos=[dx, dy, 0], rgba=rgba, mass=0.012, friction=[1.0, 0.01, 0.001], condim=4)
    rod_colors = {"red": [0.9, 0.3, 0.25, 1], "green": [0.5, 0.85, 0.3, 1], "yellow": [0.95, 0.85, 0.3, 1]}
    for i, (name, rgba) in enumerate(rod_colors.items()):
        b = world.add_body(name=f"rod_{name}", pos=[0.55, 0.1 * (i - 1), 0.01])
        b.add_freejoint(name=f"rod_{name}_free")
        b.add_geom(name=f"rod_{name}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ROD_R, ROD_HALF_LEN],
                   rgba=rgba, mass=0.008, friction=[1.0, 0.02, 0.004], condim=6)
    # Power strip: 0.42 x 0.11 x 0.04 m, USB end toward the arms; a USB-A socket cut as a slot in a front block.
    strip = world.add_body(name="strip", pos=[0.42, 0.0, 0.02])
    strip.add_freejoint(name="strip_free")
    strip.add_geom(name="strip_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.21, 0.055, 0.02], pos=[0.03, 0, 0],
                   rgba=[0.12, 0.12, 0.13, 1], mass=0.9, friction=[0.9, 0.01, 0.001])
    # USB end panel (x = -0.18 face) with one socket, low on the end so a seated plug rests flat on the table (as a
    # real plug with its cable does; a socket high on the panel left the sim plug hanging and tipping over).
    # The socket's centre is PORT_Z above the table: strip-local z = PORT_Z - 0.02. Opening 15 x 8 mm, 12 mm deep,
    # for an 11.6 x 4.2 mm tip: looser than a real socket so scripted inserts seat.
    pz0 = PORT_Z - 0.02
    for k, (py, pz, hy, hz) in enumerate(((0.0, 0.004 + 0.004, 0.0115, 0.004),  # top wall
                                          (0.0, -0.004 - 0.0008, 0.0115, 0.0008),  # thin floor above the table
                                          (0.0095, 0.0, 0.002, 0.004), (-0.0095, 0.0, 0.002, 0.004))):
        strip.add_geom(name=f"usb_port_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, hy, hz],
                       pos=[-0.186, py, pz0 + pz], rgba=[0.2, 0.2, 0.22, 1], mass=0.001)
    strip.add_site(name="usb_port", pos=[-0.186, 0, pz0], size=[0.003, 0, 0], rgba=[0, 1, 0, 0.5])
    # USB plug: metal tip 12 x 11.6 x 4.2 mm on a chunky 44 x 16 x 20 mm housing (the station's orange charging
    # cable, 44 mm with its strain relief), lying flat; starts seated in the socket (metal_sim.reset places it).
    plug = world.add_body(name="usb_plug", pos=[0.2, 0.0, PORT_Z])
    plug.add_freejoint(name="usb_plug_free")
    plug.add_geom(name="plug_tip", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.0058, 0.0021],
                  pos=[0.018, 0, 0], rgba=[0.75, 0.75, 0.78, 1], mass=0.003, friction=[0.4, 0.01, 0.001])
    # Housing plus strain relief, 44 mm long: its front face sits at the socket when seated, and the 40 mm wide
    # fingers grip its back half, clear of the socket frame.
    plug.add_geom(name="plug_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.022, 0.008, PORT_Z],
                  pos=[-0.010, 0, 0], rgba=[0.95, 0.45, 0.1, 1], mass=0.010, friction=[1.2, 0.02, 0.002],
                  condim=4)
    plug.add_site(name="plug_tip_site", pos=[0.024, 0, 0], size=[0.003, 0, 0], rgba=[1, 0, 1, 0.5])
    _add_ball_props(world)
    del spec


# Stage 3 props (the relay's "balls" stage). 40 mm balls like the station's ping-pong balls (calibration.yaml), a
# little heavier than real ones so the sim contact stays stable. Two of each coin-flip colour, two never-grab white
# ones and the black 8-ball. They start parked on the floor behind the arms; metal_sim.reset("relay") lays them out.
BALL_R = 0.020
BALLS = {"ball_orange_0": [0.98, 0.5, 0.15, 1], "ball_orange_1": [0.98, 0.5, 0.15, 1],
         "ball_green_0": [0.1, 0.8, 0.55, 1], "ball_green_1": [0.1, 0.8, 0.55, 1],
         "ball_white_0": [0.95, 0.95, 0.95, 1], "ball_white_1": [0.95, 0.95, 0.95, 1],
         "ball_black": [0.05, 0.05, 0.05, 1]}
BASKET_IN, BASKET_H, BASKET_WALL = 0.065, 0.06, 0.005  # inner half-width, wall height, wall thickness (m)


def _add_ball_props(world: mujoco.MjsBody) -> None:
    for i, (name, rgba) in enumerate(BALLS.items()):
        b = world.add_body(name=name, pos=[-0.6, -0.3 + 0.08 * i, -0.75 + BALL_R])
        b.add_freejoint(name=f"{name}_free")
        b.add_geom(name=f"{name}_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], rgba=rgba,
                   mass=0.012, friction=[1.0, 0.01, 0.002], condim=4)
    # Our bin: an open white basket (floor plus four walls), heavy enough not to slide when a ball drops in.
    basket = world.add_body(name="basket", pos=[-0.6, 0.4, -0.75 + 0.003])
    basket.add_freejoint(name="basket_free")
    white = [0.92, 0.92, 0.9, 1]
    basket.add_geom(name="basket_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BASKET_IN, BASKET_IN, 0.003],
                    rgba=white, mass=0.4)
    o, h = BASKET_IN + BASKET_WALL, BASKET_H / 2
    for k, (dx, dy, hx, hy) in enumerate(((o, 0, BASKET_WALL, o), (-o, 0, BASKET_WALL, o),
                                          (0, o, BASKET_IN, BASKET_WALL), (0, -o, BASKET_IN, BASKET_WALL))):
        basket.add_geom(name=f"basket_w{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, h],
                        pos=[dx, dy, h], rgba=white, mass=0.1)


if __name__ == "__main__":
    path = build()
    m = mujoco.MjModel.from_xml_path(str(path))
    print(f"wrote {path}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.ncam} cameras, {m.neq} equalities")
    print("cameras:", [m.camera(i).name for i in range(m.ncam)])
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])

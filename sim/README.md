# NUCK-3 Hill Climb simulation

This is a MuJoCo copy of our station: two MakerMods Metal arms (the vendor URDF), a table, the foam cubes and rods, the power strip with a USB plug, and the station's three cameras. Scripted demonstrations are recorded with the hackathon kit's own dataset writer, so the output is the same LeRobot v2.1 format `dhk-collect` writes on the real station.

Everything runs in the DimOS environment (`repos/dimos/.venv`, which has MuJoCo 3.10).

```bash
# rebuild the scene after editing build_scene.py
repos/dimos/.venv/bin/python sim/build_scene.py
# look at the cameras for a joint pose (degrees)
repos/dimos/.venv/bin/python sim/render_check.py /tmp/views 0 -30 60 -30 0 0
# record scripted demos (only successful attempts are kept)
repos/dimos/.venv/bin/python sim/record.py --task stack --episodes 20 --out sim_data/sim-stack
```

## Files

| File | What it does |
|---|---|
| `build_scene.py` | URDF to MJCF, both arms with position servos, collision groups, Dream's tool point, cameras, props. Writes `assets/scene.xml` |
| `metal_sim.py` | `MetalSim`: reset with randomized props, 30 Hz step, kit-layout `state()` and `action()`, camera `images()`, IK about the tool point, and the task checks `tower_levels()`, `usb_seated()` and `usb_unplugged()` |
| `demos.py` | Scripted demos: `Demo.stack(levels=3 or 5)` and `Demo.usb()` |
| `record.py` | Runs demos and writes the successful ones to a dataset |

`reset("relay")` lays out all three relay stages on one table: the hub in the centre, blocks on the right, seven 40 mm balls (2 orange, 2 green, 2 white, the black 8-ball) and the basket on the left. In the `stack` and `usb` tasks the balls and basket stay parked on the floor behind the arms, so those tasks get the same layouts and robot state as before (a few pixels per frame differ by physics noise). `balls()` and `in_basket()` are the stage 3 checks. The DimOS harness in `dimos_hillclimb/` drives this layout.

## Status (Nov 7)

| Piece | Status |
|---|---|
| Dataset format | Verified on the station: `tools/add_eef.py --arm metal` passes Dream's checks, and `tools/finetune.py --check` says "ready to upload" |
| Stacking, 3 pieces (blue cube, red rod, green cube) | Works: about 3 to 6 of every 8 scripted attempts succeed, and failures are discarded |
| Stacking, 5 pieces | The top piece is out of reach for a top-down placement. It needs tilted grasps |
| USB replug | Works: about 7 of every 12 attempts unplug and re-seat the plug. The re-insert is closed-loop: it lines the plug up within 1 mm of the socket axis, then pushes home while correcting from the plug's measured pose. Failures (the plug jams on the socket face) are discarded |

## Differences from the real station (read before training on this)

- **Rods start standing upright**, not lying flat as on the real table, so the demos need no re-orientation in the hand. Set `sim.rods_upright = False` for flat rods; the demos can't handle those yet.
- **Stacking uses only the right arm**, with the tower close to its base so the top stays reachable.
- **Mounting, camera poses and prop sizes are estimates** from photos (bases 0.6 m apart, 5 cm cubes, 12 cm rods). Measure the rig and update `build_scene.py`.
- **The USB socket is looser than a real one** (about 1.7 mm a side) and sits low on the strip's end, so the seated plug rests flat on the table. The plug is a chunky 44 x 16 x 20 mm housing, so the 40 mm wide fingers can grip its back half clear of the socket.
- **The scripted expert uses ground-truth poses** (where the plug and socket really are) to line up the insert. A real policy has to get that from the cameras.
- **No calibration effects or latency.** The real arms are position-servoed with gravity compensation and see 2 to 3 s Dream round trips.
- Sim images are clean renders, so Dream's fine-tune on sim-only data will not transfer well by itself. Use sim data for building and testing the harness, and mix it with real demos (or don't) after checking real-robot success.

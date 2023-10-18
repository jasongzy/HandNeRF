import argparse
import os
import sys

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="../../data/InterHand2.6M_30fps/", type=str)
    parser.add_argument("--split", default="train", type=str, help="default split when omitted in --seq")
    parser.add_argument("--capture", default=0, type=int, help="default capture_id when omitted in --seq")
    parser.add_argument("--seq", default=["0000_neutral_relaxed", "0002_good_luck", "0003_fake_gun", "0004_star_trek", "0007_thumbup_normal", "0010_thumbtuckrigid", "0011_aokay", "0013_surfer", "0014_rocker", "0016_fist", "0018_alligator_open", "0020_indexpoint", "0023_one_count", "0024_two_count", "0026_four_count", "0027_five_count", "0029_indextip", "0030_middletip", "0031_ringtip", "0032_pinkytip", "0035_palmdown", "0037_fingerspreadrelaxed", "0040_capisce", "0041_claws", "0042_pucker", "0043_peacock", "0044_cup", "0045_shakespearesyorick", "0048_index_point", "0058_middlefinger"], nargs="+")  # fmt: skip
    parser.add_argument("--new_name", default="various_poses", type=str)
    parser.add_argument("--hand_type", default="right", type=str, choices=["left", "right", "both"])
    args = parser.parse_args()

    new_path = os.path.join(args.basedir, "merge", args.new_name)
    hand_list = ["left", "right"] if args.hand_type == "both" else [args.hand_type]

    for i in range(len(args.seq)):
        sec = lambda x: len(x.strip("/").split("/"))
        if sec(args.seq[i]) == 1:
            args.seq[i] = os.path.join(f"Capture{args.capture}", args.seq[i])
        if sec(args.seq[i]) == 2:
            args.seq[i] = os.path.join(args.split, args.seq[i])

    data_root_list = list(map(lambda x: os.path.join(args.basedir, "images", x), args.seq))
    print(f"Merging the following sequences into {new_path}:")
    for data_root in data_root_list:
        assert os.path.exists(data_root), f"Wrong path: {data_root}"
        print(data_root)

    # os.system(f"rm -rf {new_path}")
    if os.path.exists(new_path):
        while True:
            choice = input(f"{new_path} already exists. rmdir? (Y/N) ").upper()
            if choice == "N":
                print("aborted")
                sys.exit(0)
            elif choice == "Y":
                os.system(f"rm -rf {new_path}")
                break
    os.makedirs(new_path, exist_ok=True)

    with open(os.path.join(new_path, "seqs.txt"), "w") as f:
        for seq in args.seq:
            f.write(f"{seq}\n")

    # frames
    frame_ids = []
    for data_root in data_root_list:
        with open(os.path.join(data_root, "frames.txt"), "r") as f:
            frames = f.readlines()
            frames = list(map(lambda x: x.strip(), frames))
            assert frames, f"no valid frame in {data_root}"
            frame_ids += frames
    with open(os.path.join(new_path, "frames.txt"), "w") as f:
        f.write("\n".join(frame_ids))

    # cams
    # os.system("cp {} {}".format(os.path.join(data_root_list[0], "cams.txt"), os.path.join(new_path, "cams.txt")))
    # with open(os.path.join(new_path, "cams.txt"), "r") as f:
    #     cam_list = f.readlines()
    #     cam_list = list(map(lambda x: "cam" + x.strip(), cam_list))
    cam_list = None
    for data_root in data_root_list:
        with open(os.path.join(data_root, "cams.txt"), "r") as f:
            cams = f.readlines()
            cams = set(map(lambda x: x.strip(), cams))
            if cam_list is None:
                cam_list = cams
            else:
                # cam_list = cam_list & cams
                assert cam_list == cams, "Cameras in different sequences are not consistent"
    cam_list = list(sorted(cam_list))
    with open(os.path.join(new_path, "cams.txt"), "w") as f:
        f.write("\n".join(cam_list))
    cam_list = list(map(lambda x: f"cam{x.strip()}", cam_list))

    # images, mask and depth
    for cam in cam_list:
        for dir in ("", "mask", "depth"):
            cam_path = os.path.join(new_path, dir, cam)
            os.makedirs(cam_path, exist_ok=True)
            for data_root in data_root_list:
                data_root = os.path.abspath(data_root)
                os.system(f"ln -s {os.path.join(data_root, dir, cam, '*')} {cam_path}")

    # params and vertices
    for dir in ("params", "vertices"):
        for hand_type in hand_list:
            path = os.path.join(new_path, dir, hand_type)
            os.makedirs(path, exist_ok=True)
            for data_root in data_root_list:
                data_root = os.path.abspath(data_root)
                os.system(f"ln -s {os.path.join(data_root, dir, hand_type, '*')} {path}")

    # annots
    cams = np.load(os.path.join(data_root_list[0], "annots.npy"), allow_pickle=True).item()["cams"]
    ims_all = []
    for data_root in data_root_list:
        ims = np.load(os.path.join(data_root, "annots.npy"), allow_pickle=True).item()["ims"]
        ims_all += ims
    annots = {"cams": cams, "ims": ims_all}
    np.save(os.path.join(new_path, "annots.npy"), annots)

    # lbs
    for hand_type in hand_list:
        lbs_root = os.path.join(new_path, "lbs", hand_type)
        os.makedirs(lbs_root, exist_ok=True)
        os.system(f"ln -s {os.path.abspath(os.path.join(data_root_list[0], 'lbs', hand_type, '*.npy'))} {lbs_root}")
        for subdir in ("bs", "bweights"):
            os.makedirs(os.path.join(lbs_root, subdir), exist_ok=True)
            for data_root in data_root_list:
                data_root = os.path.abspath(data_root)
                os.system(
                    f"ln -s {os.path.join(data_root, 'lbs', hand_type, subdir, '*')} {os.path.join(lbs_root, subdir)}"
                )

    print("Done!")

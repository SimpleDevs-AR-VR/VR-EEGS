import os
import argparse
from ExtractFrameTimestamps import ExtractTimestamps

def ProcessFootage(root, camera, mappings, vr_events, side="left", verbose=True):

    # Step 1: Build relative URLs for each of the necessary files
    camera_path = os.path.join(root, camera)
    mappings_path = os.path.join(root, mappings)
    vr_events_path = os.path.join(root, vr_events)

    # Step 2: Extract the timestamps and save them
    timestamps_path = ExtractTimestamps(camera_path, False, verbose)

    # Step 3: Correct the footage
    #cmd_str = f"ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of json {args.input}"
    #output = json.loads(subprocess.check_output(cmd_str, shell=True).decode('utf-8'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="The root directory of the user data. Must be a directory.")
    parser.add_argument('camera', help="The filename of the raw camera footage. It must be the RAW footage, aka nothing has been croppped or anything.")
    parser.add_argument("mappings", help="The filename of the mappings for eye cursor correction.")
    parser.add_argument("vr_events", help="The filename of the csv file that stores the events captured from the VR simulation app.")
    parser.add_argument("-s", "--side", help="Which eye (left or right) should we look at?", choices={"left", "right"})
    parser.add_argument('-v', '--verbose', help="Should we be verbose in printing out statements?", action="store_true")
    args = parser.parse_args()

    print(args)

    ProcessFootage(args.root, args.camera, args.mappings, args.vr_events, args.verbose)


import argparse
from realtime import realtime_emotions
from predictions import prediction_path

def run_realtime():
    realtime_emotions()
    
def runfrompath(path):
    prediction_path(path)
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_name", type=str,
                        help="Select a function to run. <realtime> or <path>")
    parser.add_argument("--path", default="saved_images/img1.jpg", type=str,
                        help="Specify the complete path where the image is saved.")
    
    args = parser.parse_args()
    
    if args.func_name == "realtime":
        run_realtime()
    elif args.func_name == "path":
        runfrompath(args.path)
    else:
        print("Usage: python main.py <function name>")
        
if __name__ == 'main':
    main()

    

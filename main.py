import argparse
from train_SRUNET import main as train_UNET  # Import the train's main function
from test import main as test_main    # Import the test's main function
from model_outputs import main as model_outputs_main  # Import the model_outputs's main function

def parse_args():
    parser = argparse.ArgumentParser(description="Super Resolution Training and Testing")

    # Full Training and Testing Pipeline
    parser.add_argument('--baseline', action='store_true', help="Flag to indicate full baseline pipeline")
    # CBSE testing pipeline
    parser.add_argument('--CBSR', action='store_true', help="Flag to indicate full CBSR pipeline")

    parser.add_argument('--dataset_path', type=str, help="Path to the dataset")


    # Arguments for traininpg
    parser.add_argument('--train', action='store_true', help="Flag to indicate training")
    parser.add_argument('--train_dataset', type=str, help="Path to training dataset")

    # Arguments for testing
    parser.add_argument('--test', action='store_true', help="Flag to indicate testing")
    parser.add_argument('--test_path', type=str, help="Path to testing dataset")
    parser.add_argument('--model_type', type=str, help="Type of the model to be tested (Baseline or CBSR)")

    # Arguments for plotting SR images
    parser.add_argument('--plot', action='store_true', help="Flag to indicate plotting")
    parser.add_argument('--images_dir', type=str, help="Path to the images directory")

    # General arguments
    parser.add_argument('--model_path', type=str, help="Path to the trained model for testing") # Path to the trained model


    return parser.parse_args()

def main():
    args = parse_args()

    if args.baseline:
        if args.dataset_path is None:
            print("Error: Dataset path is required when --full is set.")
            return
        # Call the main() from train.py
        train_UNET(args.dataset_path)
        test_main(args.dataset_path, "Best_SRUNET_4X.pth", "baseline")
        model_outputs_main(args.dataset_path, "Best_SRUNET_4X.pth")
    
    if args.CBSR:
        if args.dataset_path is None:
            print("Error: Dataset path is required when --CBSR is set.")
            return
        # Call the main() from train.py
        test_main(args.dataset_path, model_type="CBSR")

    if args.train:
        if args.train_dataset is None:
            print("Error: Training dataset path is required when --train is set.")
            return
        # Call the main() from train.py
        train_UNET(args.train_dataset)

    if args.test:
        if args.test_path is None or args.model_path is None:
            print("Error: Test dataset path and model path are required when --test is set.")
            return
        # Call the main() from test.py
        test_main(args.test_path, args.model_path, args.model_type)
    
    if args.plot:
        if args.images_dir is None or args.model_path is None:
            print("Error: Images directory and model path are required when --plot is set.")
            return
        # Call the main() from model_outputs.py
        model_outputs_main(args.images_dir, args.model_path)

if __name__ == "__main__":
    main()

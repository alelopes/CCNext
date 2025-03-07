import argparse
import models
import numpy as np
from collections import OrderedDict

def convert_dual_to_single_decoder(input_path, output_path):
    """
    Converts a dual decoder model's weights to a single decoder model's weights and saves the converted weights.
    Args:
        input_path (str): The file path to the input weights file of the dual decoder model.
        output_path (str): The file path where the converted single decoder model's weights will be saved.

    This function performs the following steps:
    1. Loads the weights of the dual decoder model from the specified input file.
    2. Initializes the dual decoder model and the single decoder model.
    3. Loads the weights into the dual decoder model.
    4. Extracts and maps the relevant weights from the dual decoder model to the single decoder model.
    5. Loads the mapped weights into the single decoder model to ensure they are in the correct format.
    6. Saves the single decoder model's weights to the specified output file.
    """
    try:
        dec_weights = torch.load(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The input file at {input_path} does not exist.")


    idep_full = models.IDEP_Skip_Dual(np.array([32, 64, 128, 256, 512]))
    idep_reduced = models.IDEP_Skip(np.array([32, 64, 128, 256, 512]))    
   
    idep_full.load_state_dict(dec_weights)

    decoders_keep_weights = [f'decoder.{i}' for i in range(13)]
    reduced_dec = OrderedDict((key, value) for key, value in dec.items() if key in idep_reduced.state_dict().keys() and '.'.join(key.split('.')[:2]) in decoders_keep_weights)

    for idx, i in enumerate(range(13, 17)):
        reduced_dec[f'decoder.{i}.conv.weight'] = dec[f'decoder.{i+10+idx}.conv.weight']
        reduced_dec[f'decoder.{i}.conv.bias'] = dec[f'decoder.{i+10+idx}.conv.bias']    

    for i in range(0, 5):
        reduced_dec[f'middle.{i}.weight'] = dec[f'middle.{i*2}.weight']    

    # Load the reduced decoder to garantee that the weights are in the correct format
    idep_reduced.load_state_dict(reduced_dec)
    
    torch.save(idep_reduced.state_dict(), output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert input dual decoder to single decoder.")
    
    # Adding arguments
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, default="reduced_icep.pt" help='Path to the output file')

    # Parsing arguments
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()

    # Accessing arguments
    input_path = args.input_path
    output_path = args.output_path

    # Print the arguments (for demonstration purposes)
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")

    convert_dual_to_single_decoder(input_path, output_path)

if __name__ == "__main__":
    main()
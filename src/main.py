import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--target-model', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--model_type', type=str, default='wikitext-2')
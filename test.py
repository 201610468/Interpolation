from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument('--netdir', type=str, nargs='+', required=False,
                        help='Path for loading the optimized network')
parser.parse_args(['--netdir', 'a'])

args = parser.parse_args()

print(args.netdir)

abc=['a',
     'b','c']
print(len(abc))






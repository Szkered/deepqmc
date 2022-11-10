from absl import app
from absl import flags
from absl import logging

from deepqmc import Molecule, evaluate, train
from deepqmc.wf import PauliNet

FLAGS = flags.FLAGS
flags.DEFINE_string("mol", "LiH", "molecule name")


def main(_):
  mol = Molecule.from_name(FLAGS.mol)
  net = PauliNet.from_hf(mol).cuda()
  train(net)
  evaluate(net)


if __name__ == '__main__':
  app.run(main)

from lambeq import BobcatParser
from discopy import grammar
from lambeq import Rewriter

sentence = 'skilful cook prepares meal'
parser = BobcatParser(verbose='text')
diagram = parser.sentence2diagram(sentence)
grammar.draw(diagram, figsize=(14,3), fontsize=12)

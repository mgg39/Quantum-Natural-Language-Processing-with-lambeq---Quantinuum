from curses import intrflush
from lambeq import BobcatParser
from discopy import grammar
from lambeq import Rewriter
from pytket.circuit.display import render_circuit_jupyter
import IPython
from lambeq import AtomicType, IQPAnsatz

#Task 3 code
sentence = 'skilful cook prepares meal'
parser = BobcatParser(verbose='text')
diagram = parser.sentence2diagram(sentence)
#grammar.draw(diagram, figsize=(14,3), fontsize=12)

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE

# Convert string diagram to quantum circuit
ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=2)
discopy_circuit = ansatz(diagram)

tket_circuit = discopy_circuit.to_tk()
circuit = render_circuit_jupyter(tket_circuit) 
print(circuit) #run windows
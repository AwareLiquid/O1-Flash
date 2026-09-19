"""mt_flash — O1-Flash: a liquid-recurrent System One decision model.

Pure typed-decision readout (Jev-style): no text generation. The recurrent
core carries an O(1) state regardless of input length; a set of typed
decision heads (Choice / Score / Probability) read probabilities from that
state in parallel.
"""

__version__ = "0.1.0"

from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

class alignment_pair(object):
	"""this would output the alignment between two pairs"""
	def __init__(self, seq1, seq2):
		super(alignment_pair, self).__init__()
		self.seq1 = seq1
		self.seq2 = seq2
		
	def alignment(self):
		a = Sequence(self.seq1)
		b = Sequence(self.seq2)
		# Create a vocabulary and encode the sequences.
		v = Vocabulary()
		aEncoded = v.encodeSequence(a)
		bEncoded = v.encodeSequence(b)

		# Create a scoring and align the sequences using global aligner.
		scoring = SimpleScoring(2, -1)
		aligner = GlobalSequenceAligner(scoring, -2)
		score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
		for encoded in encodeds:
			self.result = v.decodeSequenceAlignment(encoded)

	def get_aligned_seqs(self):
		seq1 = self.result.first
		seq2 = self.result.second
		return (''.join(seq1),''.join(seq2))


if __name__ == '__main__':
	seq1 = 'TCGTGTGAACTGCGAGGGACGCAAAGCCTCGG'
	seq2 = 'ACGTGTGAACTGCGAGGGACGCAAAGCCTCGG'
	align = alignment_pair(seq1, seq2)
	align.alignment()
	temp1, temp2 = align.get_aligned_seqs()
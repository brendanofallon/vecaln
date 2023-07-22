
## Self-supervised representation learning for DNA sequences

This repo houses code for two things:

   1. Learning embeddings for DNA sequences
   2. Using the learned embeddings with FAISS (or a similar vector search utility) to create a simple NGS short read mapper

Because we want to do 2, some requirements must be enforced on 1. Namely, we want embeddings for sequences that differ only by typical
genomic variation (like a SNP, or a small indel) to be close to sequences that don't have that variation. This way we can accurately map
sequences (reads) that contain variants to the right spot on the reference genome. An additional constraint is that we want two random sequences
to be as far as possible from one another, to help ensure the right mapping. This is basically a contrastive loss, and the good news is 
that negative examples are easy to produce - they're just random sequences. 

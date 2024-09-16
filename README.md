# Bertaphore

## Overview of projects

**Attentions in language**\
This projects aims at merging topological methods with language analysis. The application would be to reason about language using models like transformers. Making a language model interpretable could help grasp more about the language for the human as well. In this project, we study the attention of transformers.\
[Presentation](https://docs.google.com/presentation/d/1UjCcLKT5R26-X0hwlUVMwTTNGA-eYCanZza-2dt0NfE/edit#slide=id.g2eddf68fc32_0_3)


**Surprise detection**\
This project is an attempt to detect parts of a text that is surprising. It is close to anomaly detection. A part of a psychoanalyst job is to catch "anomalies". Here an anomaly is something that bears a particular interest ("parole pleine et vraie" lit. trans.: true and full speech)

**Metaphor generation**


## Motivation

<!-- intended for dialogue analysis bc simple sentences, and the text is more free-->

Human teach machines to make sense of language, machine are interpreted to understand language. We code something to better understand our own code.

## Projects and discussion

### Surprise detection

See [the notebook](notebooks/surprise-detection.ipynb)

<img src="https://github.com/PierrickLeroy/bertaphore/blob/master/images/readme_images/shark_laywer_prey.png" width="500" />

**Discussion**
The current method seem to give good results when the surprise is located in a single word. If there are other parts in the text that relate to the "surprising" vocabulary or idea, the detector does not seem to work. This is due to the fact that the detection is framed as a Masked Language Modeling task, with only one word masked at a time. This means that for an idea to be truly pivotal and therefore detected it must be condensed in one word.

**Next steps**
- [x] Single words suprise detection for simple sentences
- [ ] Test on a few corpora of dialogue
- [ ] Add logits feature to have an alternative for the rank score


### Attentions in language

**Discussion**

**Next steps**
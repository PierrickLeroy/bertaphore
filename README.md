# bertaphore
A project to detect surprise in texts.

## Motivation

<!-- intended for dialogue analysis bc simple sentences, and the text is more free-->


## Approach 

<!-- describe how is works-->

## Examples
<img src="https://github.com/PierrickLeroy/bertaphore/blob/master/images/readme_images/shark_prey.png" width="500" />

## Discussion

The current method seem to give good results when the surprise is located in a single word. If there are other parts in the text that relate to the "surprising" vocabulary or idea, the detector does not seem to work. This is due to the fact that the detection is framed as a Masked Language Modeling task, with only one word masked at a time. This means that for an idea to be truly pivotal and therefore detected it must be condensed in one word.


## Next steps

- [x] Single words suprise detection for simple sentences
- [ ] Test on a few corpora of dialogue
- [ ] Add logits feature to have an alternative for the rank score

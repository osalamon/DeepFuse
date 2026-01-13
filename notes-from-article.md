## Ad 1. Intro

- Abstract goal: "In order to make silver-standard annotations an efficient basis for gold-standard annotations, fusion methods need to level up."
- How? "CNNs have recently been very popular for image analysis tasks, including instance segmentation, due to their ability to  train specialized filters for given input images."
- IMPORTANT NOTEs: 
  1. DF is NOT ensemble!
  2. On inputs (2a) it takes RESULTS of individual raters (2b), i.e. SEGMENTATION MASKs!
  3. DF IS FUSION AND REFINEMENT NETWORK!!!!
  4. It refines segmentation of inputs (i.e. 2a) by correcting wrongly segmented pixels via learned behavior of raters (i.e. 2b) 
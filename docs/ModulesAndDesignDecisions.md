# Module List

1. Data Loader
    a. future augmentation likely goes here
2. Encoder
    a. can vary encoder models here - an interface
3. CV split
    a. decision: do we need one split for the overall training and one for the prediction head training
        - we want the refiner to see predicitions similar to those that a prediction head would create on data it has not seen previously
4. prediction head
4b. peek-in eval - evaluate metrics at this stage to then measure the efficiacy of our refiner at each steps
    - same as final evaluation - not incorporated into the training
5. patchify
6. z initialization
7. tokenizer
    a. Cache toeknized x for all splits
------- Loop Start -------
8. fuse x,y,z (agg, concat, learned combo, other)
    a. define 
9. projection of tokens to K,Q,V
10. positional rotation of K,Q
11. attention
    a. how many heads
    b. norm'ing
    c. FFN (probably just include this to attention layer)
    d. number of transformer layers
12. CNN
    a. feed the original image back in here?
    b. feed z directly in here? (I would guess not)
13. head for prediction and memory creation?
14. evaluation
    a. image, pred, mask comparison visual tool
    - not incorporated into model training until final loop is completed
------- Loop End -------


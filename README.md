# VerAs

This is the codebase for [VerAs: Verify then Assess STEM Lab Reports](https://arxiv.org/abs/2402.05224). We propose an end-to-end neural architecture (VerAs) that has separate verifier and assessment modules for a formative assessment of longer forms of student writing with rubrics. Below, you can find the figure for VerAs.

![VerAs(1)](https://github.com/psunlpgroup/VerAs/assets/29717371/0006f415-7541-4672-a2ac-0d92898c5976)


### How to run?
We experiment with two different datasets: college physics and middle school essays. You can find the commands for both below.
1. College Physics: ```python  main.py --top_k <topk> --dataset_name college_physics --verifier_model <verifier_model> --grader_model <grader_model> --loss_function <loss> --oll_loss_alpha <alpha>```
2. Middle School Essays: ```python  main.py --top_k <topk> --dataset_name middle_school --verifier_model <verifier_model> --grader_model <grader_model> --loss_function <loss> --oll_loss_alpha <alpha>```

where ```<topk>``` is a parameter that represents the number of relevant sentences retrieved by the verifier, ```<verifier_model>``` is the language model for the verifier (currently, we support bert and sbert), ```<grader_model>``` is the language model for the grader (currently, we support longt5, bert, and electra), ```<loss>``` is the loss function to use which can be cross_entropy or ordinal log loss (oll), ```<alpha>``` is the parameter for oll. 

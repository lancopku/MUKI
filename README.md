# Model Uncertainty--aware Knowledge Integration (MUKI)

[Findings of EMNLP22] From Mimicking to Integrating: Knowledge Integration for Pre-Trained Language Models

## Setup

We recommend to use virtual environment for re-producing the results.

```bash
conda create -n muki python=3.7.10
conda activate muki
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch 
pip install -r requirements.txt
```

## Train Teacher Models

The main setup of our paper is to train two teacher models specialized in different class subsets of a classification problem.

Take the THU-CNews as an example, run the following command to obtain two teacher models:

```bash
bash scripts/train_teacher.sh

```

## Knowledge Integration 

After the training of teacher model finished, we can perform knowledge intergration via various distillation methods.

```bash
# vanilla KD
bash scripts/vkd.sh 

# UHC
bash scripts/uhc.sh 

# DFA 
bash scripts/dfa.sh

# MUKI(Ours)
bash scripts/muki.sh

```

For our methods MUKI, please check the script and corresponding model file `models/uka_multiple_teacher.py` for more details.

For the Monte-Carlo dropout, to reduce the computation of uncertainty estimation, we pre-compute the scores and saved it into a numpy file (see `models/monte_carlo.py` for details). The integration can be conducted by reading the corresponding files to accelerate training.

We provide the corresponding weights in [Google Drive](https://drive.google.com/file/d/1l_p_WStrMP_zGkEp77I8cH_gvwfrd0Ya/view?usp=sharing).

It can also be achieved by compute the uncertainty on-the-fly for your own custome dataset, by adding code like below:

```python
with torch.no_grad():  # Monte Carlo Dropout on the fly
	probs = []
        for m in range(self.mc_number): # monte carlo dropout number 
        for i, t_model in enumerate(self.teachers):
        	t_model.train() # activate dropout 
		teacher_output = t_model(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask,
                                                 inputs_embeds=inputs_embeds,
                                                 output_attentions=output_attentions,
                                                 output_hidden_states=False,
                                                 return_dict=return_dict, )
                        # bsz, seq_len, logits
            teacher_logit = teacher_output[0]
            teacher_prob = F.softmax(teacher_logit, dim=-1)
                        # print(teacher_prob)
            if m == 0:
                probs.append(teacher_prob)  #
            else:
                probs[i] += teacher_prob

  
            probs = [prob / self.mc_number for prob in probs]

            # get the logits
            t_model.eval()
            for i, t_model in enumerate(self.teachers):
                teacher_output = t_model(input_ids,
                                             attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            position_ids=position_ids,
                                            head_mask=head_mask,
                                            inputs_embeds=inputs_embeds,
                                            output_attentions=output_attentions,
                                            output_hidden_states=False,
                                            return_dict=return_dict, )
                t_logit = teacher_output[0]  # bsz, seq_len, logits
                t_logits.append(t_logit)

            teacher_probs = [F.softmax(t_logit / self.kd_temperature, dim=-1) for t_logit in t_logits]
```

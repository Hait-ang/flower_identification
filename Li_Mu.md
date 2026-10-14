# 李沐
## 如何读论文——论文精读 1
- **论文的结构**：
    - title
    - abstract
    - introduction
    - method
    - experiment
    - conclusion
- **三遍法读论文**：
    - **Pass 1(write-in candidate)**: title → abstract → conclusion → diagrams and tables in experient and method → descide whether to continue reading or not
    - **Pass 2**: roughly look through the paper，figure out important diagrams and tables,underline related reference.
        - what problem does it solve, , what's the reault, what method does it use
        - if you think it's to difficult to read the paper, you can read the references to help you understand.
        - if it's unnecessary to totally understand the paper, you can stop at this step.
    - **Pass 3(the most detailed pass)**: understand every sentence and imagine the whole process in your mind.

## Alex —— part 1(论文精读 2)
### Pass 1
- **Supervised study**: there is standard answers
    - **classfication tasks** and so on.
- **Unsupervised study**: there is no standard answers.
    - **LLM** and so on.
- **The passage led to a tide in Unsupervised Study**. In this type of study, we believe that "the lager, the deeper, the better.
- In the experiment, we discover the final vector of related graphs are similar, **which prove that the Deep Neural Networks can extrat the features of the graphs and translate it into numbers**.
## Alex —— part 2(论文精选)
### Pass 2
- **You need to read lots of paper, then summery many different excellent insights from the outstanding researchers. In the end, form your own unique perspectives. It's the most important thing of every researcher.**
- most of **the first paragraph** is about **what's the research, which direction, what is in the direction, why it is important**.
- the first few paragrah is similar to a story.
- Draw features in **raw RGB**.
- **non-saturating nonlinerity funtion(ReLU) is much better and faster than saturating nonlinerity funtion**.
    - The demand of computational resource of ReLU is much less than Sigmoid/tanh(x) and so on.
    - In the diagram of ReLU, you can see there is a step change in the origin.this step change might lead to better result.
    - **In fact, ReLu is not so much better than other functions now. The main reason we usually use it is the easiness.**
- The two GPU contact from C1 to C2/C5 to FC1/FC to FC2.
    - **model parallel**.
- Because the FC of CNN is not too large, so the result of Dropout is not so good.
- **weight decacy**: mostly mean that introduce $L2$ regularization.
- **Warmup**: 
- **As green, do not pay too much attention to the experiment process. You should focus on the results of the experiment**, unless you wanna repeat it or you are an expert in the field.
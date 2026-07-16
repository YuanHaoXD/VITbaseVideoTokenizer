#### G.3. Multimodal Image Understanding

(Prompt: Generate the latex code of this formula.)
 $ A_{n} = a_{0} \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^{k} \right] $

Response:
 $$ 
\begin{aligned}
& [\\
& A\_{n}=a\_{0} \text{left}[1+\text{frac}\{3\}\{4\} \text{sum}_{\{k=1\}}\hat{\{n\}}}\\
& \text{left}(\text{frac}\{4\}\{9\} \text{right})^{\wedge}\{k\} \text{right}]\\
& ]
\end{aligned}
 $$ 

<div style="text-align: center;"><img src="imgs/img_in_image_box_102_189_1062_891.jpg" alt="Image" width="78%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_137_468_549_660.jpg" alt="Image" width="33%" /></div>


Response:
import matplotlib.pyplot as plt
labels = ['Sales', 'Tech', 'HR', 'Marketing']
sizes = [20,30,10,40]
colors = ['orange', 'blue', 'red', 'green']
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors,
    autopct='0%1.1f%%', startangle=90)
ax.axis('equal')
plt.show()

<div style="text-align: center;">Figure 14. Qualitative results on multimodal image Understanding. First, HYDRA exhibits superior proficiency in transforming complex visual data into structured text. From accurately generating LaTeX and Python code for formulas and charts to extracting fine-grained details from concept maps, the results highlight its powerful comprehension of diverse and abstract visual inputs.</div>

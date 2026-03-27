import matplotlib.pyplot as plt
import numpy as np

# 데이터
categories = ["All at Once", "Step by Step", "Propsed Method"]
values1 = [53.17, 3.17, 54.76]
values2 = [43.65, 2.38, 59.79]

x = np.arange(len(categories)) 
width = 0.3  

# 막대 그래프 그리기
plt.bar(x - width/2, values1, width, label="with out 2-stage", color="gray", edgecolor="black")
plt.bar(x + width/2, values2, width, label="with 2-stage", color="white", edgecolor="black", hatch="//")

# 축/라벨
plt.xticks(x, categories)
plt.ylabel("Agent-Level Accuracy")
plt.xlabel("Algorithm-Generated")
plt.legend()
plt.tight_layout()
plt.savefig("grouped_bar.pdf", dpi=300)
plt.show()

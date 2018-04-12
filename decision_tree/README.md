# 决策树

## 1. 基本概念

决策树是一种常见的机器学习方法,一般一棵决策树为一颗多叉树.
每一个叶子节点就对应于一个决策结果.决策树的生成过程类似于数据结构中的树的生成过程.
________________________________________
输入:

训练集$D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$

属性集$A=\{a_1,a_2,...,a_d\}$

```python
西瓜书基本算法:
过程: 函数TreeGenerate(D, A)
生成节点node
if D中样本全属于同一类别C then
    将node标记为C类叶节点; return
end if
if A为空 OR D中样本在A上的取值相同 then
    将node标记为叶节点,其分类标记为D中样本数最多的类; return
end if
从A中选择最优的划分属性a;
for a 的每一个值av do
    为node生成一个分支;令Dv表示D中在a上取值为av的样本集;
    if Dv 为空 then
        将分支标记为叶节点,其类别标记为D中样本最多的类; return
    else
        以TreeGenerate(Dv, A \ {a})为分支节点
    end if
end for
```

输出: 以node为根节点的一颗决策树
________________________________________

以下为python代码的伪代码, 参考<<机器学习实战>>

```python
def create_tree(data_set, labels):
    if D中样本全输入同一类别C:
        return 类别C
    if A为空:
        return D中样本数最多的类
    从A中选取最优划分属性a
    node = {label: {}}
    for a的每一个属性值av:
        node[label][av] = {}
        令Dv表示D在属性a上取值av的样本子集
        if Dv为空:
            node[label][av] = D中样本最多的类
        else:
            node[label][av] = create_tree(Dv, labels)
    return node
```

在<<西瓜书>>中有三种情况导致递归返回:(1)当前节点包含的样本全属于同一类别, 无需划分;(2)当前属性集为空,或是所有样本在所有属性上取值相同,无法划分;(3)当前节点包含的样本集合为空,不能划分

在<<机器学习实战>>没有判断"所有样本在所有属性上取值相同"这个条件,个人认为原因有两个: 其一, 判断的难度比较大,代码复杂,耗费时间长; 其二, 在满足条件"所有样本在所有属性上取值相同"这个条件时, 其所有样本类别有很大概率是属于同一类, 再者继续训练也只会形成一个单叉树.

## 2. 划分选择

在以上的算法流程中,最重要的步骤就是在属性集A中选择最优的划分属性a, 一般在划分的过程中,希望划分出来的样本子集尽量属于同一类别, 即节点的"纯度"越来越高.

### 2.1 信息增益

["信息熵"](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))是度量样本集合纯度最常用的一种指标. 假定当前样本集合D中第$k$类样本的比例为$p_k(k=1,2,...,|\mathcal{Y}|)$, 则$D$的信息熵定义为

$$Ent(D)= \sum ^{|\mathcal{Y}|}_{k=1}p_klog_2p_k$$

$Ent(D)$的值越小,则$D$的纯度越高.

计算信息熵时约定:若$p=0$, 则$plog_2p=0$. $Ent(D)$的最小值为0,最大值为$log_2|\mathcal{Y}|$s

下面给出信息增益的计算公式

$$Gain(D, a)=Ent(D)- \sum ^V_{v=1} \frac {|D|}{|D^v|}Ent(D^v)$$

$V\{{a^1,a^2,...,a^v}\}$为属性$a$的属性值集合; $D^v$为使用属性$a$在$D$中进行划分,$D$中属性$a$的属性值为$a^v$的样本子集;  $\frac{|D|}{|D^v|}$为每个分支节点上的权重.

一般而言, 信息增益越大, 则意味着使用属性$a$来进行划分所获得的"纯度提升"越大. 所有在算法中选择属性$a_*={arg max}_{a \in A}Gain(D, a)$

python代码[tree_gain.py](https://github.com/mejhwu/machine_learning/blob/master/decision_tree/tree_gain.py)

### 2.2 增益率

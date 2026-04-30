# 光线追踪实验项目

本项目实现了基于 Taichi 的 GPU 加速光线追踪渲染器。

## 项目结构

```
test5/
├── test5_1.py    # 基础光线追踪实现
├── test5_2.py    # 增加折射和抗锯齿
└── README.md     # 项目说明文档
```

***

## 必做部分

### 实现功能

#### 1. 三维场景搭建

| 几何体    | 位置             | 材质   | 特性      |
| ------ | -------------- | ---- | ------- |
| 无限大平面  | y = -1.0       | 漫反射  | 黑白棋盘格纹理 |
| 红色漫反射球 | (-1.2, 0.0, 0) | 漫反射  | 纯红色     |
| 银色镜面球  | (1.2, 0.0, 0)  | 镜面反射 | 理想反射    |

#### 2. 迭代式光线弹射

核心数据结构：

- `throughput`：光线能量吞吐量（衰减系数）
- `final_color`：累计最终颜色

光线追踪流程：

```
for bounce in range(max_bounces):
    1. 求交场景获取交点信息
    2. 若未击中，叠加背景色并终止
    3. 镜面材质：反射光线，更新 throughput
    4. 漫反射材质：计算光照，叠加颜色并终止
```

#### 3. 硬阴影与精度处理

**Shadow Acne 解决方案**：将射线起点沿法线方向偏移 `1e-4`

```python
ro = p + N * 1e-4  # 防止自相交
```

#### 4. UI 交互面板

提供以下滑动条控件：

- **Light X/Y/Z**：动态调整点光源位置
- **Max Bounces**：设置最大弹射次数（1-5）

### 核心代码逻辑

```python
@ti.kernel
def render():
    for i, j in pixels:
        # 生成主射线
        ro = ti.Vector([0.0, 1.0, 5.0])
        rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
        
        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        
        # 迭代式光线追踪
        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)
            # ... 材质分支处理
```
【效果演示】
<img width="800" height="656" alt="20260430_201517-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/bd198019-0c27-463a-866a-25b5de089ccf" />

***

## 选做部分 

### 1. 折射与玻璃材质 (+15%)

**实现原理**：

#### 斯涅尔定律（Snell's Law）

η 1 ​ sin θ 1 ​ = η 2 ​ sin θ 2 ​

代码实现：

```python
@ti.func
def refract(I, N, eta):
    cosi = ti.max(-1.0, ti.min(1.0, I.dot(N)))
    # ... 根据折射率计算折射方向
```

#### 菲涅尔方程（Fresnel Equation）

计算反射/折射能量比例：

```python
@ti.func
def fresnel(I, N, eta):
    # ... 计算菲涅尔反射率
    return (Rs * Rs + Rp * Rp) / 2.0
```

#### 全反射处理

当入射角大于临界角时发生全反射：

```python
if sint >= 1.0:
    result = 1.0  # 完全反射
```

**材质修改**：将红色漫反射球改为玻璃材质（IOR = 1.5）

### 2. 抗锯齿 (MSAA) (+10%)

**实现方式**：在每个像素内随机采样多次

```python
for s in range(sample_count):
    offset_x = ti.random() - 0.5  # [-0.5, 0.5)
    offset_y = ti.random() - 0.5
    # 生成带偏移的射线
    rd = normalize(ti.Vector([u + offset_x, v + offset_y, -1.0]))
    color_sum += trace_ray(ro, rd)

pixels[i, j] = color_sum / sample_count  # 颜色平均
```

**新增控件**：

- **MSAA Samples**：设置每像素采样次数（1-16）
【效果演示】
<img width="800" height="656" alt="20260430_202525-ezgif com-optimize" src="https://github.com/user-attachments/assets/b879e05c-d67a-44ff-920e-9a007ce10f23" />




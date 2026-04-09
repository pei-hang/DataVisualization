import numpy as np


def golden_section_search(phi, a, b, tol=1e-8, max_iter=200):
    """
    黄金分割搜索
    作用：
        在区间 [a, b] 内，只利用函数值，逼近一维局部最小点

    参数：
        phi      : 一维函数
        a, b     : 搜索区间左右端点
        tol      : 区间长度小于这个值时停止
        max_iter : 最大迭代次数

    返回：
        alpha    : 近似最优点
        phi(alpha): 对应函数值
    """

    # 黄金分割比例，大约 0.618
    gr = (np.sqrt(5) - 1) / 2

    # 在区间内部放两个测试点 c 和 d
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    # 计算两个测试点上的函数值
    fc = phi(c)
    fd = phi(d)

    # 最多迭代 max_iter 次
    for _ in range(max_iter):
        # 如果区间已经足够小，就停止
        if abs(b - a) < tol:
            break

        # 如果 c 点更小，说明最小值更可能在 [a, d] 内
        if fc < fd:
            b = d          # 把右端点缩到 d
            d = c          # 原来的 c 变成新的 d
            fd = fc        # 原来的 fc 变成新的 fd
            c = b - gr * (b - a)   # 重新计算新的 c
            fc = phi(c)            # 计算新的 phi(c)
        else:
            # 否则最小值更可能在 [c, b] 内
            a = c          # 把左端点缩到 c
            c = d          # 原来的 d 变成新的 c
            fc = fd        # 原来的 fd 变成新的 fc
            d = a + gr * (b - a)   # 重新计算新的 d
            fd = phi(d)            # 计算新的 phi(d)

    # 最后返回当前区间中点，作为近似最优步长
    alpha = 0.5 * (a + b)
    return alpha, phi(alpha)


def bracket_3points(phi, delta, expand=2.0, max_iter=25):
    """
    三点括区函数
    作用：
        找到 a1 < a2 < a3，使得
            phi(a1) > phi(a2) < phi(a3)
        也就是“两高一低”

    含义：
        如果一条线上出现这种关系，说明 a2 附近存在局部极小值

    参数：
        phi      : 一维函数
        delta    : 初始试探步长
        expand   : 每次扩张步长的倍数
        max_iter : 最大扩张次数

    返回：
        a1, a2, a3 : 三个点
        success    : 是否成功找到“两高一低”
    """

    # 第一个点取 0，表示不动
    a1 = 0.0
    f1 = phi(a1)

    # 第二个点取 delta，表示先沿当前方向走一步
    a2 = float(delta)
    f2 = phi(a2)

    # 如果走一步都没有下降，说明这个方向不好，直接失败
    if f2 >= f1:
        return None, None, None, False

    # 如果确实下降了，就继续沿这个方向往前扩
    for _ in range(max_iter):
        # 第三个点取 a2 的 expand 倍
        a3 = a2 * expand
        f3 = phi(a3)

        # 如果出现“左高、中低、右高”，说明括区成功
        if f3 > f2:
            return a1, a2, a3, True

        # 否则说明函数还在继续下降，需要继续往前扩
        # 把原来的 a2 当成新的 a1
        a1, f1 = a2, f2
        # 把原来的 a3 当成新的 a2
        a2, f2 = a3, f3

    # 如果扩了很多次还没找到，就返回失败
    return None, None, None, False


def normalize(v):
    """
    把向量归一化成单位向量
    作用：
        只保留方向，不保留长度
    """

    n = np.linalg.norm(v)   # 向量长度
    if n < 1e-15:           # 如果长度太小，防止除零
        return v
    return v / n            # 返回单位向量


def build_directions(n, num_random=8, rng=None):
    """
    构造搜索方向集

    方向包括两部分：
    1. 坐标方向：±e1, ±e2, ..., ±en
    2. 随机方向：随机生成若干方向，再单位化

    参数：
        n           : 变量维数
        num_random  : 随机方向的数量
        rng         : 随机数生成器

    返回：
        dirs : 方向列表，每个元素都是一个方向向量
    """

    if rng is None:
        rng = np.random.default_rng()

    dirs = []

    # ---------- 生成坐标方向 ----------
    for i in range(n):
        e = np.zeros(n)   # 先生成全零向量
        e[i] = 1.0        # 第 i 个位置设为 1，得到第 i 个基向量
        dirs.append(e)    # 正方向
        dirs.append(-e)   # 负方向

    # ---------- 生成随机方向 ----------
    for _ in range(num_random):
        v = rng.normal(size=n)   # 生成一个随机向量
        v = normalize(v)         # 单位化，只保留方向
        if np.linalg.norm(v) > 0:
            dirs.append(v)       # 加入正方向
            dirs.append(-v)      # 加入负方向

    return dirs


def dfo_localmin_mixed_dirs(fun, x0, max_iter=500, tol_step=1e-10,
                            delta0=0.2, shrink=0.7, expand=2.0,
                            max_bracket_iter=25, line_tol=1e-8,
                            line_max_iter=200, num_random_dirs=8,
                            verbose=True, seed=0):
    """
    无导数局部极小搜索算法（混合方向版）

    思想：
        1. 在当前点生成一批方向（坐标方向 + 随机方向）
        2. 试探每个方向是否能下降
        3. 如果能下降，就沿这个方向做一维搜索
        4. 用“三点两高一低”先找到局部谷底所在区间
        5. 再用黄金分割在区间里细找最优步长
        6. 如果这一轮所有方向都不行，就缩小步长

    参数：
        fun               : 目标函数，只需要能计算函数值
        x0                : 初始点
        max_iter          : 最大迭代次数
        tol_step          : 步长小于这个值就停止
        delta0            : 初始试探步长
        shrink            : 步长缩小比例
        expand            : 括区时的扩张倍数
        max_bracket_iter  : 三点括区最大扩张次数
        line_tol          : 一维搜索精度
        line_max_iter     : 一维搜索最大迭代次数
        num_random_dirs   : 每轮额外加入的随机方向数
        verbose           : 是否打印过程
        seed              : 随机种子

    返回：
        x       : 最终点
        fx      : 最终函数值
        history : 迭代历史
    """

    # 把初始点转成一维浮点数组
    x = np.array(x0, dtype=float).reshape(-1)

    # 变量维数
    n = len(x)

    # 固定随机种子，保证每次运行随机方向一致
    rng = np.random.default_rng(seed)

    # 当前试探步长
    delta = float(delta0)

    # 当前函数值
    fx = fun(x)

    # 用于记录历史，方便后面画图分析
    history = {
        "x": [x.copy()],       # 每一轮的点
        "f": [fx],             # 每一轮的函数值
        "delta": [delta],      # 每一轮的步长
    }

    # 开始主迭代
    for k in range(1, max_iter + 1):
        if verbose:
            print(f"Iter {k:3d}: f = {fx:.12e}, delta = {delta:.4e}")

        # 如果步长已经非常小，说明再搜也很难有明显变化，停止
        if delta < tol_step:
            break

        # improved 表示这一轮有没有找到更优点
        improved = False

        # 先把当前点看作这一轮的“最好点”
        best_x = x.copy()
        best_f = fx

        # 构造本轮要尝试的方向集合
        dirs = build_directions(n, num_random=num_random_dirs, rng=rng)

        # 依次尝试每一个方向
        for d in dirs:
            # 先沿方向 d 走一步 delta，得到试探点
            xt = x + delta * d

            # 计算试探点函数值
            ft = fun(xt)

            # 如果这个方向走一步就能下降，说明值得继续沿这个方向深挖
            if ft < best_f:
                # 定义沿方向 d 的一维函数
                # a 表示沿这个方向走多远
                phi = lambda a, x=x, d=d: fun(x + a * d)

                # 先找三个点，满足“两高一低”
                a1, a2, a3, success = bracket_3points(
                    phi, delta, expand=expand, max_iter=max_bracket_iter
                )

                if success:
                    # 如果成功找到括区，就在 [a1, a3] 内精细搜索最优步长
                    alpha, _ = golden_section_search(
                        phi, a1, a3, tol=line_tol, max_iter=line_max_iter
                    )
                else:
                    # 如果括区失败，就保守地直接走一步 delta
                    alpha = delta

                # 计算按这个步长走过去后的候选点
                xcand = x + alpha * d
                fcand = fun(xcand)

                # 如果这个候选点比当前轮已知最好点更优，就更新
                if fcand < best_f:
                    best_x = xcand
                    best_f = fcand
                    improved = True

        # ---------- 一轮方向都试完以后 ----------
        if improved:
            # 如果这一轮找到更优点，就真正更新当前位置
            x = best_x
            fx = best_f
        else:
            # 如果这一轮所有方向都没有改进，说明当前步长可能太大
            # 缩小步长，再继续尝试
            delta *= shrink

        # 把本轮结果记录下来
        history["x"].append(x.copy())
        history["f"].append(fx)
        history["delta"].append(delta)

    return x, fx, history


if __name__ == "__main__":
    def rastrigin(x):
        x = np.asarray(x)
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    # 初始点
    x0 = np.array([1.2, -1])

    # 调用无导数优化算法
    x_star, f_star, hist = dfo_localmin_mixed_dirs(
        rastrigin,
        x0,
        max_iter=600,         # 最大迭代次数
        tol_step=1e-12,       # 步长小于这个阈值时停止
        delta0=0.2,           # 初始步长
        shrink=0.7,           # 步长缩小比例
        expand=2.0,           # 三点括区时的扩张倍数
        max_bracket_iter=25,  # 括区最大次数
        line_tol=1e-10,       # 一维搜索精度
        line_max_iter=300,    # 一维搜索最大迭代次数
        num_random_dirs=16,   # 每轮随机方向数量
        verbose=True,         # 打印迭代过程
        seed=42               # 固定随机种子，方便复现实验
    )

    # 打印最终结果
    print("\n最终点:")
    print(x_star)
    print("最终函数值:")
    print(f_star)
============================================Generating Model============================================
b0
Without Pre-trained model
w, d, s, p 1.0 1.0 224 0.2
[BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[2], se_ratio=0.25), 
BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 
BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 
BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 
BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25),
BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 
BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)]

GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2, image_size=224, include_top=True)

b5
Without Pre-trained model
w, d, s, p 1.6 2.2 456 0.4
BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)
GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.4, num_classes=1000, width_coefficient=1.6, depth_coefficient=2.2, depth_divisor=8, min_depth=None, drop_connect_rate=0.2, image_size=456, include_top=True)

Without Pre-trained model
w, d, s, p 1.0 1.0 112 0.2
BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)
BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25)
BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)
GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2, image_size=112, include_top=True)


finish getmodel
[NCE]: params Z -1, Z_momentum 0.9, tau 0.1
==========================================!!!START TRAINING!!!==========================================
-------
torch.Size([90, 1, 16, 112, 112])
torch.Size([90, 1, 17, 113, 113])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([32, 1, 3, 3, 3]) None (2, 2, 2) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 32, 8, 56, 56])
torch.Size([90, 32, 9, 57, 57])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([32, 1, 3, 3, 3]) None [2, 2, 2] (0, 0, 0) (1, 1, 1) 32
-------
torch.Size([90, 32, 1, 1, 1])
torch.Size([90, 32, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([8, 32, 1, 1, 1]) Parameter containing:
tensor([-0.0176,  0.1004,  0.1057, -0.0453, -0.1588,  0.1411,  0.0689, -0.1190],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 8, 1, 1, 1])
torch.Size([90, 8, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([32, 8, 1, 1, 1]) Parameter containing:
tensor([ 0.2980, -0.3372,  0.0063, -0.0170, -0.3041, -0.2746,  0.2517,  0.2706,
         0.1363, -0.1241,  0.1661,  0.2837,  0.2381, -0.3491,  0.3168,  0.0489,
        -0.2159,  0.1290, -0.1458, -0.1782,  0.1029,  0.0977,  0.1831, -0.2357,
         0.1068,  0.2503,  0.2878,  0.0243, -0.1222, -0.2212, -0.2661, -0.2453],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 32, 4, 28, 28])
torch.Size([90, 32, 4, 28, 28])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([16, 32, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 16, 4, 28, 28])
torch.Size([90, 16, 4, 28, 28])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([96, 16, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 96, 4, 28, 28])
torch.Size([90, 96, 5, 29, 29])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([96, 1, 3, 3, 3]) None [2, 2, 2] (0, 0, 0) (1, 1, 1) 96
-------
torch.Size([90, 96, 1, 1, 1])
torch.Size([90, 96, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([4, 96, 1, 1, 1]) Parameter containing:
tensor([ 8.4765e-05,  2.3267e-02,  7.7162e-02, -2.2358e-03], device='cuda:0',
       requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 4, 1, 1, 1])
torch.Size([90, 4, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([96, 4, 1, 1, 1]) Parameter containing:
tensor([-0.2676, -0.4580,  0.2157,  0.3083, -0.3323,  0.4832, -0.1737, -0.1867,
         0.1987,  0.0264,  0.3350, -0.4949, -0.3212,  0.2648,  0.0389, -0.0159,
         0.0056,  0.1388, -0.1030, -0.2368,  0.2424, -0.2923,  0.1133,  0.4973,
        -0.3858, -0.0981,  0.3445, -0.4988, -0.2238,  0.0526,  0.4907, -0.0015,
        -0.2009, -0.2189,  0.0312,  0.0072, -0.3521,  0.0776,  0.4042,  0.1528,
        -0.2962, -0.4951,  0.3498, -0.1197,  0.2416,  0.2840,  0.0225,  0.4181,
        -0.2882, -0.1684, -0.2799, -0.2391, -0.4123,  0.3820, -0.2434,  0.1515,
        -0.1389,  0.3033, -0.0806,  0.2987,  0.1610,  0.1370,  0.1762,  0.0422,
         0.3117, -0.3859, -0.2655,  0.0764, -0.2117, -0.1491,  0.0494,  0.1401,
         0.2650,  0.2050,  0.2461,  0.3985,  0.1025,  0.1890, -0.4304, -0.0526,
         0.4334, -0.2285,  0.2357, -0.2095, -0.2707, -0.4457, -0.3607, -0.3834,
        -0.1882,  0.1841,  0.1129, -0.4351,  0.0611, -0.4164,  0.2606,  0.2906],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 96, 2, 14, 14])
torch.Size([90, 96, 2, 14, 14])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([24, 96, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 24, 2, 14, 14])
torch.Size([90, 24, 2, 14, 14])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 24, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 144, 2, 14, 14])
torch.Size([90, 144, 4, 16, 16])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 1, 3, 3, 3]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 144
-------
torch.Size([90, 144, 1, 1, 1])
torch.Size([90, 144, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([6, 144, 1, 1, 1]) Parameter containing:
tensor([-0.0790,  0.0166,  0.0279, -0.0612, -0.0740, -0.0069], device='cuda:0',
       requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 6, 1, 1, 1])
torch.Size([90, 6, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 6, 1, 1, 1]) Parameter containing:
tensor([ 0.3380,  0.2180,  0.2256,  0.3969,  0.2965,  0.2219,  0.0365, -0.2366,
        -0.2108,  0.0907,  0.0352, -0.1062,  0.0671,  0.0856,  0.0517, -0.0285,
         0.0570,  0.2826,  0.1308,  0.1944,  0.0432,  0.0572,  0.1028, -0.2901,
         0.0082, -0.1142,  0.3295, -0.3446, -0.2002,  0.2442, -0.0398,  0.3903,
        -0.2446, -0.0193,  0.3877,  0.2662, -0.0779,  0.2333, -0.3133, -0.3475,
        -0.1178, -0.2227, -0.2830,  0.4004,  0.2678,  0.3309, -0.0676, -0.3781,
        -0.3143, -0.1115, -0.3707, -0.2458, -0.4065, -0.2545, -0.0186, -0.1672,
        -0.3121,  0.2479, -0.0964,  0.2991, -0.0580,  0.2263,  0.2994,  0.2225,
         0.0518,  0.1152, -0.0930, -0.1799, -0.1727, -0.3111, -0.3000, -0.2250,
         0.3388,  0.0160, -0.2488,  0.2388, -0.1377, -0.0168, -0.2944,  0.1049,
         0.0685,  0.1485, -0.2948, -0.0219,  0.4029, -0.1957, -0.2028,  0.0950,
        -0.0680,  0.3184,  0.2587, -0.1682, -0.0996, -0.3951,  0.1242,  0.0403,
        -0.2964,  0.1258, -0.2681,  0.3965,  0.0253,  0.0740,  0.0616, -0.1251,
         0.0936, -0.1532,  0.3674,  0.0291,  0.2591,  0.1342,  0.0933, -0.3660,
        -0.0190,  0.2248,  0.1648, -0.1368, -0.2338, -0.3315, -0.3725,  0.3861,
         0.3896,  0.4010, -0.3377,  0.1620,  0.2105,  0.1864, -0.3939,  0.2846,
        -0.1423, -0.1600,  0.1556,  0.1886,  0.0749,  0.1524,  0.2999, -0.0446,
         0.3838,  0.2809, -0.1197, -0.0402, -0.0477, -0.0998,  0.0073,  0.2721],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 144, 2, 14, 14])
torch.Size([90, 144, 2, 14, 14])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([24, 144, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 24, 2, 14, 14])
torch.Size([90, 24, 2, 14, 14])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 24, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 144, 2, 14, 14])
torch.Size([90, 144, 5, 17, 17])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 1, 5, 5, 5]) None [2, 2, 2] (0, 0, 0) (1, 1, 1) 144
-------
torch.Size([90, 144, 1, 1, 1])
torch.Size([90, 144, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([6, 144, 1, 1, 1]) Parameter containing:
tensor([ 0.0306, -0.0622, -0.0428, -0.0814, -0.0723,  0.0433], device='cuda:0',
       requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 6, 1, 1, 1])
torch.Size([90, 6, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([144, 6, 1, 1, 1]) Parameter containing:
tensor([-0.1375, -0.3202,  0.3921, -0.1487, -0.2680,  0.0571,  0.1836, -0.2975,
        -0.0666, -0.3173,  0.3042, -0.2718, -0.3903, -0.2947, -0.0223, -0.4043,
        -0.1555, -0.3714, -0.2279,  0.2911,  0.0839,  0.3889, -0.0792,  0.2564,
         0.1173, -0.0404,  0.2607, -0.1787,  0.0033,  0.0885,  0.3812,  0.4027,
        -0.3582,  0.0416, -0.0722,  0.0870, -0.2237, -0.1057, -0.3598,  0.0432,
        -0.1966,  0.2974,  0.1820,  0.3912, -0.1988, -0.1237,  0.3656,  0.0253,
        -0.2451,  0.1861, -0.2639,  0.2743,  0.0781,  0.3096,  0.2487, -0.0881,
         0.0989,  0.0358,  0.0525, -0.1912, -0.2808, -0.1081, -0.0893,  0.2961,
         0.0247,  0.1117, -0.3797,  0.1922, -0.3953, -0.0500, -0.2978, -0.0961,
         0.3776,  0.1483,  0.1756, -0.2727, -0.0964, -0.0814,  0.2451, -0.1325,
         0.0541,  0.0312,  0.2828,  0.2685, -0.0142, -0.1558,  0.1864,  0.2289,
        -0.2730, -0.1414,  0.2195, -0.2731, -0.1046,  0.3580,  0.3019, -0.0223,
        -0.2277, -0.2785, -0.0043,  0.3304,  0.1881,  0.2931, -0.1353, -0.3313,
         0.4037, -0.0217, -0.1353, -0.2755, -0.2748, -0.2965, -0.0817,  0.0658,
        -0.2468, -0.0328, -0.1539,  0.0487, -0.3244,  0.3631, -0.2267,  0.0311,
        -0.2725, -0.1073,  0.0269, -0.0394,  0.0483, -0.0446,  0.2949,  0.2820,
         0.0592,  0.3834, -0.1367,  0.3726, -0.1238,  0.2715, -0.0741, -0.3478,
        -0.3662,  0.3405, -0.1930,  0.0304,  0.2038,  0.3407,  0.3172, -0.2144],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 144, 1, 7, 7])
torch.Size([90, 144, 1, 7, 7])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([40, 144, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 40, 1, 7, 7])
torch.Size([90, 40, 1, 7, 7])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([240, 40, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 240, 1, 7, 7])
torch.Size([90, 240, 5, 11, 11])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([240, 1, 5, 5, 5]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 240
-------
torch.Size([90, 240, 1, 1, 1])
torch.Size([90, 240, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([10, 240, 1, 1, 1]) Parameter containing:
tensor([ 0.0011,  0.0169, -0.0474,  0.0605, -0.0450, -0.0366,  0.0594, -0.0353,
        -0.0382,  0.0336], device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 10, 1, 1, 1])
torch.Size([90, 10, 1, 1, 1])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([240, 10, 1, 1, 1]) Parameter containing:
tensor([ 0.1914,  0.2591,  0.2613,  0.0108, -0.3010, -0.2294, -0.0357, -0.0414,
        -0.2591,  0.1966,  0.2846,  0.1332, -0.1058,  0.1913,  0.1497,  0.1418,
        -0.1694, -0.0559,  0.2957, -0.1704, -0.3018, -0.2344,  0.1011, -0.1008,
         0.3135,  0.0143,  0.2720,  0.1197,  0.1428, -0.0233,  0.1362,  0.0964,
         0.0294,  0.2274, -0.2363, -0.2771,  0.1059,  0.1929,  0.1856, -0.2786,
        -0.1268,  0.2661,  0.1787, -0.2443,  0.2213,  0.2046,  0.0680,  0.2913,
         0.3008, -0.3059,  0.0924,  0.2471, -0.2775,  0.2509,  0.1003, -0.2510,
         0.0721,  0.3015, -0.0316,  0.0723, -0.0860, -0.2316,  0.0513, -0.0810,
         0.0896,  0.3095, -0.0550, -0.2530, -0.2808, -0.1285, -0.1078, -0.1203,
        -0.1183,  0.2005, -0.0458,  0.1882,  0.2008, -0.2601, -0.0181, -0.0393,
        -0.2047, -0.3159, -0.2609, -0.0929, -0.0721, -0.2448, -0.1056,  0.2936,
         0.0937, -0.1866, -0.2658, -0.1679,  0.2055, -0.0430, -0.0081,  0.1385,
        -0.2975,  0.0382, -0.0108,  0.0623, -0.3107,  0.1357, -0.2070,  0.0021,
        -0.1257, -0.2565,  0.1473,  0.1512,  0.1880,  0.0064, -0.2454, -0.1747,
         0.1076,  0.2450, -0.1091, -0.1675,  0.1581,  0.2840, -0.1188, -0.1375,
        -0.3057,  0.1498,  0.1443, -0.0917,  0.0043,  0.2352,  0.2074,  0.0303,
         0.2683, -0.1497, -0.0160, -0.2071,  0.0115, -0.0161,  0.1546, -0.2685,
        -0.2066, -0.2019,  0.1111, -0.2945, -0.1420,  0.3101, -0.3141, -0.1761,
        -0.1844,  0.2249,  0.1556,  0.0188, -0.2575, -0.2514,  0.1542, -0.1767,
         0.1482, -0.1995, -0.0793,  0.1181, -0.1775, -0.1063,  0.1039, -0.2909,
         0.1880,  0.1750,  0.1590, -0.0925, -0.0557, -0.2738,  0.0922, -0.3110,
         0.2754, -0.1839,  0.2506, -0.2166,  0.0603,  0.1568,  0.1888, -0.0499,
        -0.0319, -0.1215,  0.1844, -0.2577,  0.2732,  0.0784, -0.3108, -0.0441,
         0.0375,  0.2741,  0.2557, -0.2718, -0.0724, -0.2792, -0.1805,  0.2109,
         0.0537,  0.2490,  0.0710, -0.0758, -0.1585,  0.1976, -0.0537,  0.0103,
        -0.1198, -0.2715,  0.2709, -0.0769, -0.2393, -0.1635, -0.1116, -0.0007,
         0.1595,  0.0148,  0.0353,  0.0651, -0.1834,  0.0746, -0.0178, -0.0956,
         0.2349,  0.2530,  0.1606,  0.0940, -0.0797, -0.2482,  0.0473,  0.0393,
         0.0529, -0.0690, -0.0872,  0.3026,  0.1513,  0.1341, -0.0317,  0.0671,
         0.0591, -0.0556,  0.2767,  0.2737,  0.1734, -0.1161,  0.0262,  0.2354],
       device='cuda:0', requires_grad=True) (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 240, 1, 7, 7])
torch.Size([90, 240, 1, 7, 7])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([40, 240, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 40, 1, 7, 7])
torch.Size([90, 40, 1, 7, 7])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([240, 40, 1, 1, 1]) None (1, 1, 1) (0, 0, 0) (1, 1, 1) 1
-------
torch.Size([90, 240, 1, 7, 7])
torch.Size([90, 240, 2, 8, 8])
[self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups] torch.Size([240, 1, 3, 3, 3]) None [2, 2, 2] (0, 0, 0) (1, 1, 1) 240
Traceback (most recent call last):
  File "main.py", line 344, in <module>
    criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)
  File "main.py", line 106, in train
    unnormed_vec, normed_vec = model(data)
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 150, in forward
    return self.module(*inputs[0], **kwargs[0])
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/efficientnet.py", line 191, in forward
    x = self.extract_features(inputs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/efficientnet.py", line 180, in extract_features
    x = block(x, drop_connect_rate=drop_connect_rate)
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/efficientnet.py", line 77, in forward
    x = self._swish(self._bn1(self._depthwise_conv(x)))
  File "/home/ubuntu/anaconda3/envs/torch140/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/ubuntu/codes/driver_monitor/Driver-Anomaly-Detection/models/utils.py", line 158, in forward
    x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
RuntimeError: Calculated padded input size per channel: (2 x 8 x 8). Kernel size: (3 x 3 x 3). Kernel size can't be greater than actual input size

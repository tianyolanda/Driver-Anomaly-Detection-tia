if frame_num == 50:
    plt.plot(frame_all, sim_all)
    plt.xlabel('Frame ID')
    plt.ylabel('Similarity score')
    x_major_locator = MultipleLocator(25)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.2)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim([1, 50])
    plt.ylim([-1, 1])

    # plt.show()
    plt.savefig('./frame-similarity.png')
    exit()

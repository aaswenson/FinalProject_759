import subprocess
import matplotlib.pyplot as plt


def const_vol(command):
    Np = 500

    # Set constant dimension.
    N = []
    dim = 1e-11
    for i in range(1,21,2):
        dim *= i
        N.append(i*i*i)

    # Set mesh size.
    list_arg = []
    for i in range(1,21,2):
        list_arg.append((i, dim/float(i)))

    # Run command.
    seq_t = []
    par_t_all = []
    par_t_com = []
    for (Nd, dvol) in list_arg:
        run_out = subprocess.run([command, str(Np), str(Nd), str(dvol)], stdout=subprocess.PIPE)
        print(str(run_out.stdout)[2:].split('\\n')[:3])
        par_t_com.append(float(str(run_out.stdout)[2:].split('\\n')[0]\
                .split('   ')[1]))
        par_t_all.append(float(str(run_out.stdout)[2:].split('\\n')[1]\
                .split('   ')[1]))
        seq_t.append(float(str(run_out.stdout)[2:].split('\\n')[2]\
                .split('   ')[1]))

    fig, ax = plt.subplots()
    ax.set_title('Timing comparison between Sequential and Parallel method.')
    ax.set_ylabel('Wall clock time [ms]')
    ax.set_xlabel('Total number of voxels')
    ax.semilogy()
    ax.plot(N, seq_t, marker='o', color='blue', label='Sequential method')
    ax.plot(N, par_t_all, marker='o', color='red', label='Parallel method, w/ mem trans')
    ax.plot(N, par_t_com, marker='o', color='green', label='Parallel method, w/o mem trans')
    ax.legend()
    plt.savefig('const_vol.png')
    print('Saved const_vol.png.')


def vary_tracks_mem(command):
    Nd = 15
    dvol = 10

    # Set ntracks.
    ntracks = []
    for i in range(1,10):
        ntracks.append(2**i)

    par_t_all = []
    par_t_com = []
    for Np in ntracks:
        run_out = subprocess.run([command, str(Np), str(Nd), str(dvol)], stdout=subprocess.PIPE)
        print(str(run_out.stdout)[2:].split('\\n')[:3])
        par_t_com.append(float(str(run_out.stdout)[2:].split('\\n')[0]\
                .split('   ')[1]))
        par_t_all.append(float(str(run_out.stdout)[2:].split('\\n')[1]\
                .split('   ')[1]))

    mem_tr = []
    for i in range(len(par_t_all)):
        mem_tr.append((par_t_all[i] - par_t_com[i])/par_t_all[i]*100)

    fig, ax = plt.subplots()
    ax.set_title('Effect of particle numbers on percentage memory process.')
    ax.set_ylabel('Percent memory process [%]')
    ax.set_xlabel('Number of particles [#]')
    ax.set_xscale('log', basex=2)
    ax.plot(ntracks, mem_tr, marker='o', color='black', label='Percent memory process')
    ax.legend()
    plt.savefig('ntracks_mem.png')
    print('Saved ntracks_mem.png.')

def vary_tracks(command):
    Nd = 15
    dvol = 10

    # Set ntracks.
    ntracks = []
    for i in range(1,10):
        ntracks.append(2**i)

    seq_t = []
    par_t_all = []
    par_t_com = []
    for Np in ntracks:
        run_out = subprocess.run([command, str(Np), str(Nd), str(dvol)], stdout=subprocess.PIPE)
        print(str(run_out.stdout)[2:].split('\\n')[:3])
        par_t_com.append(float(str(run_out.stdout)[2:].split('\\n')[0]\
                .split('   ')[1]))
        par_t_all.append(float(str(run_out.stdout)[2:].split('\\n')[1]\
                .split('   ')[1]))
        seq_t.append(float(str(run_out.stdout)[2:].split('\\n')[2]\
                .split('   ')[1]))

    fig, ax = plt.subplots()
    ax.set_title('Timing comparison between Sequential and Parallel method.')
    ax.set_ylabel('Wall clock time [ms]')
    ax.set_xlabel('Total number of voxels')
    ax.semilogy()
    ax.plot(ntracks, seq_t, marker='o', color='blue', label='Sequential method')
    ax.plot(ntracks, par_t_all, marker='o', color='red', label='Parallel method, w/ mem trans')
    ax.plot(ntracks, par_t_com, marker='o', color='green', label='Parallel method, w/o mem trans')
    ax.legend()
    plt.savefig('ntracks.png')
    print('Saved ntracks.png.')


def mfp():
    ntracks = 20
    Nd = 15
    dvol = 10

    # Set ntracks.
    mfp       = [0.1, 0.3, 0.6, 0.9, 3, 6, 9, 30, 60, 90]
    par_t_com = [29647, 2192.57, 564.785, 233.592,
                 26.794, 7.20794, 4.25062, 0.436224, 0.200704, 0.15872]
    par_t_all = [29693.1, 2198.94, 567.806, 235.354,
                 27.8558, 7.93834, 4.97056, 1.13328, 0.889408, 0.847552]
    seq_t     = [447.111, 37.2972, 10.2799,
                 4.14045, 0.622368, 0.223136, 0.140288, 0.027264, 0.016384, 0.0168]

    fig, ax = plt.subplots()
    ax.set_title('Effect of mean free path on timing.')
    ax.set_ylabel('Wall clock time [ms]')
    ax.set_xlabel('Mean free path [cm]')
    ax.semilogy()
    ax.plot(mfp, seq_t, marker='o', color='blue', label='Sequential method')
    ax.plot(mfp, par_t_all, marker='o', color='red', label='Parallel method, w/ mem trans')
    ax.plot(mfp, par_t_com, marker='o', color='green', label='Parallel method, w/o mem trans')
    ax.legend()
    plt.savefig('mfp.png')
    print('Saved mfp.png.')

if __name__=="__main__":
    exec_comm = './par_tally'
    const_vol(exec_comm)
    vary_tracks_mem(exec_comm)
    vary_tracks(exec_comm)
    mfp()

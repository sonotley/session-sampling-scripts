"""Quick script to generate true and resampled session timelines

"""

from matplotlib import pyplot as plt

from session_sampling_simulator import session_simulator as sim

DURATION = 1000
SAMPLE_PERIOD = 50

query = sim.Query(1,10,25,{1:1}, 40, "lognormal")

session = sim.generate_session(window_duration=DURATION, queries=[query])//10

st = sim.get_sample_times(
        session_duration=DURATION,
        sample_period=SAMPLE_PERIOD,
        strategy=sim.SamplingStrategy.UNIFORM,
    )

sample = session[st]

fig, axs = plt.subplots(2, 1)

axs[0].bar(height=session%10>0, x=range(DURATION), width=1, color="#fe5286")

axs[1].bar(height=sample%10>0, x=range(0, DURATION, SAMPLE_PERIOD), width=SAMPLE_PERIOD, color="#fe5286")

for i in range(2):
    axs[i].set_xlim((0,DURATION))
    axs[i].set_ylim((0,1))
    axs[i].set_xticks(range(0, DURATION, SAMPLE_PERIOD))
    axs[i].grid(axis='x')
    axs[i].set_xticklabels([])
    axs[i].set_yticks([0.5])

    axs[i].tick_params(axis="both", length=0)

axs[0].set_yticklabels([sum(session%10>0)/DURATION])
axs[1].set_yticklabels([SAMPLE_PERIOD * sum(sample%10>0)/DURATION])

plt.show()
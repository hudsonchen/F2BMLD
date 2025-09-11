# Define the command template
command_template = (
    "--policy_noise_level {policy_noise} "
    "--noise_level {noise} "
    "--lagrange_reg {lagrange} "
    "--seed {seed} "
    "--max_steps {steps} "
    "--batch_size 32"
)

noise = 0.3

# Open the file toy_local.sh in write mode
with open(f'/home/zongchen/F2BMLD/scripts/myriad/configs_noise_{noise}.txt', 'w') as file:
    # Generate commands for seeds 0 to 100 and write each to the file
    for seed in range(5):
        for policy_noise in [0.0, 0.1]:
                for lagrange in [0.1, 0.3]:
                    for steps in [500_000]:
                        command = command_template.format(
                            seed=seed,
                            policy_noise=policy_noise,
                            noise=noise,
                            lagrange=lagrange,
                            steps=steps
                        )
                        file.write(command + "\n")

# Make the script executable (works on Unix-based systems)
import os
os.chmod(f'/home/zongchen/F2BMLD/scripts/myriad/configs_noise_{noise}.txt', 0o755)

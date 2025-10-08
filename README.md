
**Purpose:**  
These metrics enable automatic dataset assessment and correlation with imitation learning performance (e.g., BC/BC-RNN policies trained using Robomimic).

---

## 🚀 Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including tutorials and guides.

Key links:
- [Installation Steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement Learning Overview](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available Environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)
- [Synthetic Data Generation (MimicGen)](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/index.html)

---

## ⚙️ Isaac Sim Version Dependency

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

> The `feature/isaacsim_5_0` branch contains active updates and may include breaking changes until the official Isaac Lab 2.2 release.

---

## 🤝 Contributing

We welcome contributions from the community to improve Isaac Lab.  
Please see the [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

For contributions related to this extended version (Filtering and DQS), please check:
- `isaaclab_mimic/generation/filtering/`
- `isaaclab_mimic/evaluation/dqs/`

---

## 🧩 Research Contributions

This repository includes work conducted as part of the **Master’s Thesis of Anshul Chauhan** at **RWTH Aachen University**.  
His research focuses on improving **synthetic data generation** and **quality evaluation** for imitation learning using Isaac Lab and MimicGen.

**Thesis focus:**  
> "Evaluation of Synthetic Data Generation Methods for Imitation Learning in Robotic Arm Manipulation Tasks."

Key research outcomes integrated here:
- Real-time **smoothness-based demonstration filtering**
- Dataset-level **Demonstration Quality Score (DQS)** system
- Quantitative evaluation pipeline for dataset-to-policy performance correlation

---

## 🧭 Support

- Use [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for questions or new ideas  
- Use [Issues](https://github.com/isaac-sim/IsaacLab/issues) for reporting bugs or tracking feature requests  

---

## 🧑‍💻 Connect

Interested in collaborating or learning more?  
Reach out via [LinkedIn](https://www.linkedin.com/in/anshul-chauhan-rwth/) or open a discussion on GitHub!

You can also join the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to connect with the NVIDIA Isaac community.

---

## 📜 License

The Isaac Lab framework is released under the [BSD-3 License](LICENSE).  
The `isaaclab_mimic` extension and its standalone scripts are released under [Apache 2.0](LICENSE-mimic).  
All dependency and asset licenses are listed in the [`docs/licenses`](docs/licenses) directory.

---

## 📚 Acknowledgement

Isaac Lab development originated from the [Orbit](https://isaac-orbit.github.io/) framework.

If you use this work in academic publications, please cite the original paper:


# SLM-A*: Fine-Tuned Small Language Models for Risk-Aware Multi-UAV Path Planning

## 📌 Overview

Multi-UAV path planning in contested airspace requires generating collision-free, risk-aware trajectories under tight computational and latency constraints.
This repository presents SLM-A*, a hybrid language-to-waypoints planning framework that fine-tunes small language models (SLMs) to predict discrete multi-UAV routes from structured mission descriptions and then verifies and repairs them using a unified geometric safety checker.
 
<img width="8680" height="4950" alt="figures-Page-10" src="https://github.com/user-attachments/assets/51440701-40f4-46c2-89da-95efcd147557" />

## 🧠 Key Idea

* Language models propose → geometry verifies → safety repairs

SLM-A* decouples semantic reasoning from hard safety constraints:

* A fine-tuned SLM predicts waypoint sequences from text-based mission descriptions.

* A geometric checker validates and repairs the trajectories to ensure safety and feasibility.

## ✨ Contributions

* Hybrid LLM-guided multi-UAV planner with deterministic safety guarantees

Fine-tuning of small language models (1–3B) using QLoRA (4-bit NF4)

Unified geometric safety checker enforcing:

* Polygonal obstacle avoidance

* Threat-field risk line integrals

* Inter-agent minimum separation

* Shared time parameterization

* Large-scale synthetic dataset of threat-aware multi-UAV scenarios

* State-of-the-art performance across mission-relevant planning metrics

* Validation in Gazebo–ArduPilot simulations with realistic autopilot execution


## 🛩️ Dataset

* 3,000 generated scenarios

* 3 UAVs, 4 threat fields, 1 mission objective

* 40 × 40 grid world

Ground truth paths generated using prioritized, time-expanded A* with:

* Clearance penalties

* Turn penalties

## 🧪 Models Evaluated

The following SLMs were adapted using QLoRA:

* Gemma-2B

* Fox-1-1.6B

* Qwen2.5-1.5B

* Phi-2-2.7B

* StableLM-3B

* TinyLlama-1.1B (selected as SLM-A*)

TinyLlama-1.1B achieved the best text-level metrics and is adopted as the proposed method.

## 📊 Evaluation Metrics

SLM-A* is evaluated on a 315-scenario held-out benchmark using:

* Makespan

* Sum of Costs (SOC)

* Number of collisions

* Risk exposure (line-integral cost)

* Minimum inter-agent separation

## 🚀 Deployment Modes

SLM-A* supports two deployment strategies:

* On-Device Inference

* Jetson-class embedded hardware

* Low-latency, decentralized execution

* Centralized Serving

* vLLM backend

* Multi-UAV client support

## 🧪 Simulation Validation

* Simulator: Gazebo

* Autopilot: ArduPilot SITL

* Control Interface: MAVROS

* Confirmed feasibility of executing SLM-generated waypoints on a realistic autopilot stack


## ⚙️ Requirements

* Python ≥ 3.9

* PyTorch

* Hugging Face Transformers

* PEFT / QLoRA

* vLLM (optional, for centralized inference)

* ROS (Noetic)

* Gazebo

* ArduPilot + MAVROS

## 🤝 Contact

* For questions, collaboration, or issues, please  contact the authors.
* email : afaq@jbnu.ac.kr

# VASU-LM Project Instructions

Model recommendation before every task

Before beginning any VASU-LM task, always state which model should be used forthat task and why, based on its difficulty and risk.

Use this recommendation policy:

GPT-5.6 Sol — architecture decisions, major refactors, difficultdebugging, checkpoint/data compatibility, training-stability reviews, andfinal technical audits.

GPT-5.6 Terra — normal coding, training scripts, evaluation tooling,experiments, and documentation.

GPT-5.6 Luna — simple or repetitive work such as formatting, logparsing, basic checks, and straightforward test generation.

GPT-5.5 — independent second opinions or comparisons against previousdecisions and results.

The response must begin with a short recommendation such as:

Recommended model: GPT-5.6 Terra — this is a routine evaluation-toolingtask with moderate implementation complexity.

Then proceed with the task. If the current interface cannot switch models,clearly say that the recommendation is advisory and continue using theavailable model.

This rule applies to coding, debugging, training, architecture, research,documentation, evaluation, and project-management tasks related to VASU-LM.
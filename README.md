# clay-project-template
Project template to run DDI research experiments.

To run experiments, I recommend using the `bocas` package to help keep your code modular, clean, and lean.  This project template shows you how to use `bocas` and use other best practices for development (especially using SMU's Maneframe).  

The intent of this repo is to have consistent repos throughout research projects and to make your code more reproducible.  This template should make your life easier by being able to clone and get off the ground running quickly.  A great thing about this template and using `bocas` is that it will autogenerate tables and results for you to compare different configurations.  And you can add these results directly to your papers!

This README is great for getting you started for your next project, but make sure to remove and adjust this README for when you publish your project-specific GitHub repo!  Outside of SMU student's, no one will want to see your Maneframe files and explanations for example.  


## Run Setup

In order to use your code as a package, cd into your `/code_package` directory and run:

```bash
python setup.py develop
```

## Gitignore Setup
Since this is a project template, I removed some items from the `.gitignore` file so 
that they are still tracked on this git repo.  To configure `.gitignore` for this 
project after getting the code on your machine, I recommend adding the following lines 
to your `.gitignore` file:

```bash
*playground
```

This is discussed further in [Directory Information](#directory-info).

If you also plan on using SMU's maneframe to run some experiments, I suggest adding the 
additional lines to your `.gitignore` file (discussed further in [Use on SMU's Maneframe](#smu-maneframe)):

```bash
RSYNC_README.md
*.out
*.sh
.superpod_ignore
```

Once you add to your `/.gitignore` file, for your first commit, run the following to remove these files from your git tracked files.  This way, they will not show up on git anymore.

```bash
git rm -r --cached .
git add .
git commit -m "Removing ignored files"
git push 
```

## <a name="directory-info"></a> Directory Information

Outling the directory structure and what each directory is used for.

### Code Package

The `/code_package` directory is where your custom code should go.  Think of this as publishable code for pip installs.  This should be a package (with modules) so that you can load the package into `/experiments` to run your experiments with your custom code!  

NOTE: `/code_package` should be modular and **not experiment specific**.  This code should be reusable for all experiments.  For example, say you have created a custom convolutional layer.  This can be applied on images, audio, etc.  This code goes in `/code_package`.  For one of your experiments, you want to test out your super cool code on audio data.  For this audio task, you come up with a custom model configuration where you stack 2 of your custom layers.  This code **does not** go in `/code_package`.  Since this is specific to the audio task, this code would go in `/experiments/audio`.  More on this [below](#experiments).

### <a name="experiments"></a> Experiments

`/experiments` is where you test out your code on a specific task.  There is a readme in the `/experiments/experiment_a` directory.  Look [there](/experiments/experiment_a/README.md) for more information on how to setup an experiment.

### <a name="playground"></a> Playground

We all make messy code, and we all need a place to run some trial and error testing to get stuff up and running...but no one else wants to see that rat's nest :-).  This is why we have `/playground`!  Add messy jupyter notebooks, scripts, and anything else you may want without subjecting others to the craziness/madness that goes into your programming!  `/code_package`, `/examples`, and `/experiments` should all have production quality code that you can share with others.  `/playground` is exactly what it sounds like, so do what you need to here, but leave it all on your machine!  

Make sure to add `*playground` to your `.gitignore` file after cloning this repo!

### Examples

`/examples` allows you to showcase your custom code to people who are interested in your research.  `/experiments` is great for your publications and making your work reproducible, but `/examples` should be more generic and not for a specific task.  

For example, if you created a custom neural network layer or just made a great python package for people to use, you could create a simple jupyter notebook in `/experiments` to show off your code and how to use it.  

Some people may be interested in the results from your publication.  They would look in `/experiments` for more details.  Other people may look at your publication and want to repurpose your code for their specific task.  `/examples` is a great place to show them how to use your code in a generic sense.

# Simon ARC Lab - My solution for ARC Prize 2024

In the ARC Prize 2024 contest, my solver got `score=1` out of 100. 
My approach uses `RLE` compression to fit in an entire puzzle into a tiny 1024 context length.
I'm new to LLMs and I'm surprised that I managed to solve 1 of the hidden tasks with this approach.

## Examples of incorrect predictions

![image](metadata/2b01abd0_incorrect.png)
![image](metadata/3f23242b_incorrect.png)
![image](metadata/a2fd1cf0_incorrect.png)
![image](metadata/cb227835_incorrect.png)

## RLE compression of an image

Given this image

```text
4 4 4 4
4 4 4 4
4 4 4 4
4 7 4 7
4 7 4 7
4 7 4 7
8 8 2 4
8 8 2 4
8 8 2 4
7 4 2 4
7 4 2 4
7 4 2 4
```

Here is the RLE representation

```text
4 12 4,,,4747,,,a824,,,7424,,
```

The `4 12` means `width=4` and `height=12`.

## RLE compression of an ARC puzzle

```text
I0
6 2 9a3b9,071b3
O0
6 7 9,d93,c939,a9a3a9,931b9,97c9,0d9
I1
3 1 4a7
O1
3 3 a97,979,4a9
I2
3 3 370,7a0,a50
O2
3 5 a90,970,3a0,759,5a9
I3
3 5 181,1a8,238,138,
O3
3 7 a91,9a8,1a8,138,238,139,1a9
I4T
3 5 0a4,494,934,a49,4
O4T
3 7 a94,9a4,694,439,9a4,a49,4a9
```

The `I0`, here `I` means input, and `0` means that it's pair 0. The input image for pair 0.

The `O0`, here `O` means output, and `0` means that it's pair 0. The output image for pair 0.

The `I4T`, here `T` means that it's the `test` pair, that is to be solved.

## Reflections on what went wrong

**Catastrophic forgetting**, Whenever I taught the model a new kind of task, then it was able to solve that kind of task, but got worse at solving other kinds of tasks.
Future plan: Explore approaches that are more robust against catastrophic forgetting.

**Previous prediction being ignored**, When solving a puzzle the model got invoked a few times. Feeding the previous prediction into the next iteration. However the model seemed to get confused about the previous prediction despite being very close to the target output. Leaving out the previous best prediction was resulting in better predictions. I had hoped that the model was better at taking a hint and refine on it, however taking hints was unsuccessful.

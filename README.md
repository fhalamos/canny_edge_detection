implement basic convolution, denoising, and edge detection operations. Your implementation should be restricted to using low-level primitives in numpy (e.g. you may not call a Python library routine for convolution in the implementation of your convolution function).

Note: ONLY Python3 supported. hw1.py is the ONLY file you need to modify and submit. 

See hw1.py for a description of the functions you must implement.  You may submit the assignment by uploading your completed version of hw1.py.

A Python script for visualization your implementation on some provided examples images is included as hw1_visualization.py.   

We also release a hw1_reference.py and self_checker.py for unit testing usage only. (hw1_reference is a external Python library impl. which produce the expected outputs and self_checker contains some very simple cases which  helps you compare your own impl. outputs and the expected outputs). You do not need to modify/submit these example test scripts. 

We provide the sample images for edge detection, which is used for evaluating the performance  on real data.  These images are located at data/edge_img and  categorized by difficulties. We also provide the synthetic image "checker.png" and the reference output of our own implemented canny edge (located at data/edge_ref) to visually debug and tune the threshold strategy.  You will get full credit if the results visually look good on easy splitting and be assigned extra credit based on the performance on harder cases.

Note that while they test some basic functionality, you should test your own solution implementations more extensively.

Grading is based on the correctness and efficiency of your implementation of the following algorithms.  See comments in hw1.py for a detailed description of each. 
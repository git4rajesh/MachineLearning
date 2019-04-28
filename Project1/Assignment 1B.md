1. What are Channels and Kernels (according to EVA)?

Channels:
=========
Channels are the primary features of an image.
How many numbers are used to specify the color of each pixel is the number of channels each pixel has.
Incase of a colored image, we have 3 channels such as Red, Green and Blue.

A monochrome image that has one number per pixel has one channel.

Feature Maps and Channels in CNN are one and the same. Each channel after the first layer of a CNN is a feature map. 
Before the first layer of CNN, RGB images have 3 channels (red, green & blue channels).

Code snippet explaining Channels:
---------------------------------

Conv2d(3, 32, kernel_size=3, stride=2, padding=2)

The 3 is the number of input channels (R, G, B). 
That 32 is the number of channels (i.e. feature maps) in the output of the first convolution operation. 
So, the first conv layer takes a color (RGB) image as input, applies 3x3 kernel with a stride 2, and outputs 32 feature maps


Kernels:
=========
a. In the context of convolutional neural networks, 
				kernel = filter = feature detectors. 
				
b. Features  like straight edges, simple colors, and curves are identified.
c. Each filter can be thought of as storing a  template/pattern. When we convolve this filter across the corresponding input, we are basically trying to find out the similarity between the stored template and different locations in the input.

d. A filter is represented by a vector of weights with which we convolve the input. They are learned and fine tuned using the Backpropagation Algorithm.

Code snippet explaining Kernel convolution:
------------------------------------------

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
	model.add(Conv2D(10, (1, 1), activation='relu'))


In the above snippet we have 3 * 3 kernel and 1 * 1 kernel respectively.

============================================================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. Why should we only (well mostly) use 3x3 Kernels?
	
	Need for odd shape for Kernel:
	------------------------------
			We generally opt for odd sized kernels like a. 3 * 3 b. 5 *5 or 7 * 7.
		The rationale is that, the odd size kernels can symmetrically demarcate the input images into left and right.
		Say for example an Image of a Triangle. a 3 * 3 Kernel can effectively convolve over it without any wastage.
		So we opt for odd sized kernels.
		
		
	Need for 3 * 3 Kernel over other Odd sized Kernels:
	--------------------------------------------------
			The 3 * 3 Kernel is optimized to convolve over any input image.
			They use less parameters/ weights compared to 5 * 5 or 7 * 7 Kernels.
			
			For example, lets says we have an 7 * 7 input image.
			
			case i: With 3 * 3 Kernel
			-------------------------
			5 * 5 (Image) ---> 3 * 3 (Kernel) ---> 3 * 3 (Image) ---> 3 * 3(Kernel) ---> 1 * 1
			
			So in this case all we need is 18 Parameters to reach the Global Receptive Field.
			
			
			case ii: With 5 * 5 Kernel
			--------------------------
			5 * 5 (Image) ---> 5 * 5 (Kernel) ---> 1 * 1 
			
			In this case we need 25 Parameters to reach the Global Receptive Field.
			
			
			case iii: With 7 * 7 Kernel
			----------------------------
			Too big a Kernel size to accommodate all sizes of images. 
			So cant be used for small sized images.
			
			
	=========================================================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3. How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

199 * 199 ----> 3 * 3 (kernel) ---> 197 * 197 ---> 3 * 3 (kernel) ---> 195 * 195 ---> 3 * 3(kernel) ---> 193 * 193
193 * 193 ----> 3 * 3 (kernel) ---> 191 * 191 ---> 3 * 3 (kernel) ---> 189 * 189 ---> 3 * 3(kernel) ---> 187 * 187
187 * 187 ----> 3 * 3 (kernel) ---> 185 * 185 ---> 3 * 3 (kernel) ---> 183 * 183 ---> 3 * 3(kernel) ---> 181 * 181
181 * 181 ----> 3 * 3 (kernel) ---> 179 * 179 ---> 3 * 3 (kernel) ---> 177 * 177 ---> 3 * 3(kernel) ---> 175 * 175

175 * 175 ----> 3 * 3 (kernel) ---> 173 * 173 ---> 3 * 3 (kernel) ---> 171 * 171 ---> 3 * 3(kernel) ---> 169 * 169
169 * 169 ----> 3 * 3 (kernel) ---> 167 * 167 ---> 3 * 3 (kernel) ---> 165 * 165 ---> 3 * 3(kernel) ---> 163 * 163
163 * 163 ----> 3 * 3 (kernel) ---> 161 * 161 ---> 3 * 3 (kernel) ---> 159 * 159 ---> 3 * 3(kernel) ---> 157 * 157
157 * 157 ----> 3 * 3 (kernel) ---> 155 * 155 ---> 3 * 3 (kernel) ---> 153 * 153 ---> 3 * 3(kernel) ---> 151 * 151
151 * 151 ----> 3 * 3 (kernel) ---> 149 * 149 ---> 3 * 3 (kernel) ---> 147 * 147 ---> 3 * 3(kernel) ---> 145 * 145

145 * 145 ----> 3 * 3 (kernel) ---> 143 * 143 ---> 3 * 3 (kernel) ---> 141 * 141 ---> 3 * 3(kernel) ---> 139 * 139
139 * 139 ----> 3 * 3 (kernel) ---> 137 * 137 ---> 3 * 3 (kernel) ---> 135 * 135 ---> 3 * 3(kernel) ---> 133 * 133
133 * 133 ----> 3 * 3 (kernel) ---> 131 * 131 ---> 3 * 3 (kernel) ---> 129 * 129 ---> 3 * 3(kernel) ---> 127 * 127
127 * 127 ----> 3 * 3 (kernel) ---> 125 * 125 ---> 3 * 3 (kernel) ---> 123 * 123 ---> 3 * 3(kernel) ---> 121 * 121
121 * 121 ----> 3 * 3 (kernel) ---> 119 * 119 ---> 3 * 3 (kernel) ---> 117 * 117 ---> 3 * 3(kernel) ---> 115 * 115

115 * 115 ----> 3 * 3 (kernel) ---> 113 * 113 ---> 3 * 3 (kernel) ---> 111 * 111 ---> 3 * 3(kernel) ---> 109 * 109
109 * 109 ----> 3 * 3 (kernel) ---> 107 * 107 ---> 3 * 3 (kernel) ---> 105 * 105 ---> 3 * 3(kernel) ---> 103 * 103
103 * 103 ----> 3 * 3 (kernel) ---> 101 * 101 ---> 3 * 3 (kernel) ---> 99 * 99 ---> 3 * 3(kernel) ---> 97 * 97
97 * 97 ----> 3 * 3 (kernel) ---> 95 * 95 ---> 3 * 3 (kernel) ---> 93 * 93 ---> 3 * 3(kernel) ---> 91 * 91
91 * 91 ----> 3 * 3 (kernel) ---> 89 * 89 ---> 3 * 3 (kernel) ---> 87 * 87 ---> 3 * 3(kernel) ---> 85 * 85

85 * 85 ----> 3 * 3 (kernel) ---> 83 * 83 ---> 3 * 3 (kernel) ---> 81 * 81 ---> 3 * 3(kernel) ---> 79 * 79
79 * 79 ----> 3 * 3 (kernel) ---> 77 * 77 ---> 3 * 3 (kernel) ---> 75 * 75 ---> 3 * 3(kernel) ---> 73 * 73
73 * 73 ---> 3 * 3 (kernel) ---> 71 * 71 ----> 3 * 3 (kernel) ---> 69 * 69 ---> 3 * 3 (kernel) ---> 67 * 67 ---> 3 * 3(kernel) ---> 65 * 65


65 * 65 ----> 3 * 3 (kernel) ---> 63 * 63 ---> 3 * 3 (kernel) ---> 61 * 61 ---> 3 * 3(kernel) ---> 59 * 59
59 * 59 ----> 3 * 3 (kernel) ---> 57 * 57 ---> 3 * 3 (kernel) ---> 55 * 55 ---> 3 * 3(kernel) ---> 53 * 53
53 * 53 ---> 3 * 3 (kernel) ---> 51 * 51 ----> 3 * 3 (kernel) ---> 49 * 49 ---> 3 * 3 (kernel) ---> 47 * 47 ---> 3 * 3(kernel) ---> 45 * 45

 
45 * 45 ----> 3 * 3 (kernel) ---> 43 * 43 ---> 3 * 3 (kernel) ---> 41 * 41 ---> 3 * 3(kernel) ---> 39 * 39
39 * 39 ----> 3 * 3 (kernel) ---> 37 * 37 ---> 3 * 3 (kernel) ---> 35 * 35 ---> 3 * 3(kernel) ---> 33 * 33
33 * 33 ----> 3 * 3 (kernel) ---> 31 * 31 ---> 3 * 3 (kernel) ---> 29 * 29 ---> 3 * 3(kernel) ---> 27 * 27
27 * 27 ----> 3 * 3 (kernel) ---> 25 * 25 ---> 3 * 3 (kernel) ---> 23 * 23 ---> 3 * 3(kernel) ---> 21 * 21
21 * 21 ----> 3 * 3 (kernel) ---> 19 * 19 ---> 3 * 3 (kernel) ---> 17 * 17 ---> 3 * 3(kernel) ---> 15 * 15

15 * 15 ----> 3 * 3 (kernel) ---> 13 * 13 ---> 3 * 3 (kernel) ---> 11 * 11 ---> 3 * 3(kernel) ---> 9 * 9
9 * 9 ----> 3 * 3 (kernel) ---> 7 * 7 ---> 3 * 3 (kernel) ---> 5 * 5 ---> 3 * 3(kernel) ---> 3 * 3
3 * 3 ----> 3 * 3 (kernel) ---> 1 * 1

Total Count of Convolution Operation = 100
Ü.
ñ

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.22v2.8.2-0-g2ea19cbb5758°Í*

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameSGD/dense/kernel/momentum

-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¾
value³B¯ B§

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
í
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
Ú
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27*

C0
D1*
* 
°
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Jserving_default* 
* 
* 
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
ó
Player-0
Qlayer_with_weights-0
Qlayer-1
Rlayer_with_weights-1
Rlayer-2
Slayer-3
Tlayer_with_weights-2
Tlayer-4
Ulayer_with_weights-3
Ulayer-5
Vlayer-6
Wlayer_with_weights-4
Wlayer-7
Xlayer_with_weights-5
Xlayer-8
Ylayer_with_weights-6
Ylayer-9
Zlayer-10
[layer_with_weights-7
[layer-11
\layer_with_weights-8
\layer-12
]layer_with_weights-9
]layer-13
^layer-14
_layer_with_weights-10
_layer-15
`layer_with_weights-11
`layer-16
alayer_with_weights-12
alayer-17
blayer-18
clayer-19
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses*
¦

Ckernel
Dbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*
\
	pdecay
qlearning_rate
rmomentum
siterCmomentuméDmomentumê*
Ú
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27*

C0
D1*
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
¬

)kernel
*bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

+kernel
,bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

-kernel
.bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

/kernel
0bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses*

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 
¬

1kernel
2bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses*
¬

3kernel
4bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses*
¬

5kernel
6bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses*

¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses* 
¬

7kernel
8bias
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses*
¬

9kernel
:bias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses*
¬

;kernel
<bias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses*

Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses* 
¬

=kernel
>bias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses*
¬

?kernel
@bias
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses*
¬

Akernel
Bbias
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses*

é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses* 

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses* 
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*
* 
* 

C0
D1*

C0
D1*
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdecay?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUElearning_rateGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEmomentumBlayer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUESGD/iter>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*

0
1*

ÿ0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

+0
,1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

-0
.1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

/0
01*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 
* 
* 

10
21*
* 
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*
* 
* 

30
41*
* 
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses*
* 
* 

50
61*
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses* 
* 
* 

70
81*
* 
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses*
* 
* 

90
:1*
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses*
* 
* 

;0
<1*
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses* 
* 
* 

=0
>1*
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses*
* 
* 

?0
@1*
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses*
* 
* 

A0
B1*
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses* 
* 
* 
Ê
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25*

P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17
b18
c19*
* 
* 
* 
* 
* 
* 
* 
* 
<

àtotal

ácount
â	variables
ã	keras_api*
M

ätotal

åcount
æ
_fn_kwargs
ç	variables
è	keras_api*

)0
*1*
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0
.1*
* 
* 
* 
* 

/0
01*
* 
* 
* 
* 
* 
* 
* 
* 
* 

10
21*
* 
* 
* 
* 

30
41*
* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 

70
81*
* 
* 
* 
* 

90
:1*
* 
* 
* 
* 

;0
<1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1*
* 
* 
* 
* 

?0
@1*
* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
hb
VARIABLE_VALUEtotalIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEcountIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

à0
á1*

â	variables*
jd
VARIABLE_VALUEtotal_1Ilayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_1Ilayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ä0
å1*

ç	variables*

VARIABLE_VALUESGD/dense/kernel/momentum_variables/26/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUESGD/dense/bias/momentum_variables/27/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
p
serving_default_bytesPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_bytesblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/bias*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_157839
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOpConst*3
Tin,
*2(	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_159522
ß
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/biasdecaylearning_ratemomentumSGD/itertotalcounttotal_1count_1SGD/dense/kernel/momentumSGD/dense/bias/momentum*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_159646Î(


H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

&__inference_dense_layer_call_fn_159052

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_155498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
ú
&__inference_model_layer_call_fn_156875	
bytes!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity	

identity_1¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallbytesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_156751k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes

c
G__inference_block4_pool_layer_call_and_return_conditional_losses_159303

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv3_layer_call_fn_159282

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
Á
,map_while_random_flip_left_right_true_157925b
^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityy
/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
*map/while/random_flip_left_right/ReverseV2	ReverseV2^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency8map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
)map/while/random_flip_left_right/IdentityIdentity3map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ý
é
6model_lambda_map_while_random_flip_up_down_true_154307v
rmodel_lambda_map_while_random_flip_up_down_reversev2_model_lambda_map_while_random_flip_up_down_control_dependency7
3model_lambda_map_while_random_flip_up_down_identity
9model/lambda/map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ¸
4model/lambda/map/while/random_flip_up_down/ReverseV2	ReverseV2rmodel_lambda_map_while_random_flip_up_down_reversev2_model_lambda_map_while_random_flip_up_down_control_dependencyBmodel/lambda/map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
3model/lambda/map/while/random_flip_up_down/IdentityIdentity=model/lambda/map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"s
3model_lambda_map_while_random_flip_up_down_identity<model/lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó
µ
)map_while_random_flip_up_down_true_156073\
Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityv
,map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
'map/while/random_flip_up_down/ReverseV2	ReverseV2Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency5map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
&map/while/random_flip_up_down/IdentityIdentity0map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ü
ö
F__inference_sequential_layer_call_and_return_conditional_losses_155699

inputs&
vgg16_155640:@
vgg16_155642:@&
vgg16_155644:@@
vgg16_155646:@'
vgg16_155648:@
vgg16_155650:	(
vgg16_155652:
vgg16_155654:	(
vgg16_155656:
vgg16_155658:	(
vgg16_155660:
vgg16_155662:	(
vgg16_155664:
vgg16_155666:	(
vgg16_155668:
vgg16_155670:	(
vgg16_155672:
vgg16_155674:	(
vgg16_155676:
vgg16_155678:	(
vgg16_155680:
vgg16_155682:	(
vgg16_155684:
vgg16_155686:	(
vgg16_155688:
vgg16_155690:	
dense_155693:	
dense_155695:
identity¢dense/StatefulPartitionedCall¢vgg16/StatefulPartitionedCallè
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_155640vgg16_155642vgg16_155644vgg16_155646vgg16_155648vgg16_155650vgg16_155652vgg16_155654vgg16_155656vgg16_155658vgg16_155660vgg16_155662vgg16_155664vgg16_155666vgg16_155668vgg16_155670vgg16_155672vgg16_155674vgg16_155676vgg16_155678vgg16_155680vgg16_155682vgg16_155684vgg16_155686vgg16_155688vgg16_155690*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_155165
dense/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0dense_155693dense_155695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_155498u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ì
ð
+__inference_sequential_layer_call_fn_158406

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155505o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv3_layer_call_fn_159212

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
Õ
Ý
4lambda_map_while_random_flip_left_right_false_157562o
klambda_map_while_random_flip_left_right_identity_lambda_map_while_random_flip_left_right_control_dependency4
0lambda_map_while_random_flip_left_right_identityè
0lambda/map/while/random_flip_left_right/IdentityIdentityklambda_map_while_random_flip_left_right_identity_lambda_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"m
0lambda_map_while_random_flip_left_right_identity9lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯
_
C__inference_CLASSES_layer_call_and_return_conditional_losses_156346

inputs
identity	R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :a
ArgMaxArgMaxinputsArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentityArgMax:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
m
B__inference_lambda_layer_call_and_return_conditional_losses_158095

inputs
identity¢	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖ_
map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ä
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs%map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
map_while_body_157865*!
condR
map_while_cond_157864*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààR
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2
	map/while	map/while:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
Þ
F__inference_sequential_layer_call_and_return_conditional_losses_158576

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	M
1vgg16_block2_conv2_conv2d_readvariableop_resource:A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv1_conv2d_readvariableop_resource:A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	M
1vgg16_block3_conv2_conv2d_readvariableop_resource:A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv3_conv2d_readvariableop_resource:A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	M
1vgg16_block4_conv1_conv2d_readvariableop_resource:A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	M
1vgg16_block4_conv2_conv2d_readvariableop_resource:A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	M
1vgg16_block4_conv3_conv2d_readvariableop_resource:A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	M
1vgg16_block5_conv1_conv2d_readvariableop_resource:A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	M
1vgg16_block5_conv2_conv2d_readvariableop_resource:A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	M
1vgg16_block5_conv3_conv2d_readvariableop_resource:A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	7
$dense_matmul_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢)vgg16/block1_conv1/BiasAdd/ReadVariableOp¢(vgg16/block1_conv1/Conv2D/ReadVariableOp¢)vgg16/block1_conv2/BiasAdd/ReadVariableOp¢(vgg16/block1_conv2/Conv2D/ReadVariableOp¢)vgg16/block2_conv1/BiasAdd/ReadVariableOp¢(vgg16/block2_conv1/Conv2D/ReadVariableOp¢)vgg16/block2_conv2/BiasAdd/ReadVariableOp¢(vgg16/block2_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv1/BiasAdd/ReadVariableOp¢(vgg16/block3_conv1/Conv2D/ReadVariableOp¢)vgg16/block3_conv2/BiasAdd/ReadVariableOp¢(vgg16/block3_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv3/BiasAdd/ReadVariableOp¢(vgg16/block3_conv3/Conv2D/ReadVariableOp¢)vgg16/block4_conv1/BiasAdd/ReadVariableOp¢(vgg16/block4_conv1/Conv2D/ReadVariableOp¢)vgg16/block4_conv2/BiasAdd/ReadVariableOp¢(vgg16/block4_conv2/Conv2D/ReadVariableOp¢)vgg16/block4_conv3/BiasAdd/ReadVariableOp¢(vgg16/block4_conv3/Conv2D/ReadVariableOp¢)vgg16/block5_conv1/BiasAdd/ReadVariableOp¢(vgg16/block5_conv1/Conv2D/ReadVariableOp¢)vgg16/block5_conv2/BiasAdd/ReadVariableOp¢(vgg16/block5_conv2/Conv2D/ReadVariableOp¢)vgg16/block5_conv3/BiasAdd/ReadVariableOp¢(vgg16/block5_conv3/Conv2D/ReadVariableOp¢
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
£
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¹
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¤
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

5vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Â
#vgg16/global_average_pooling2d/MeanMean"vgg16/block5_pool/MaxPool:output:0>vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense/MatMulMatMul,vgg16/global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

Á
-map_while_random_flip_left_right_false_156027a
]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityÓ
)map/while/random_flip_left_right/IdentityIdentity]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ù
e
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158703

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv3_layer_call_and_return_conditional_losses_159293

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
¤
-__inference_block2_conv1_layer_call_fn_159122

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs
¾	
æ
lambda_map_while_cond_1571462
.lambda_map_while_lambda_map_while_loop_counter-
)lambda_map_while_lambda_map_strided_slice 
lambda_map_while_placeholder"
lambda_map_while_placeholder_12
.lambda_map_while_less_lambda_map_strided_sliceJ
Flambda_map_while_lambda_map_while_cond_157146___redundant_placeholder0
lambda_map_while_identity

lambda/map/while/LessLesslambda_map_while_placeholder.lambda_map_while_less_lambda_map_strided_slice*
T0*
_output_shapes
: 
lambda/map/while/Less_1Less.lambda_map_while_lambda_map_while_loop_counter)lambda_map_while_lambda_map_strided_slice*
T0*
_output_shapes
: y
lambda/map/while/LogicalAnd
LogicalAndlambda/map/while/Less_1:z:0lambda/map/while/Less:z:0*
_output_shapes
: g
lambda/map/while/IdentityIdentitylambda/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "?
lambda_map_while_identity"lambda/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


H__inference_block3_conv1_layer_call_and_return_conditional_losses_159183

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs


H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

û
F__inference_sequential_layer_call_and_return_conditional_losses_155943
vgg16_input&
vgg16_155884:@
vgg16_155886:@&
vgg16_155888:@@
vgg16_155890:@'
vgg16_155892:@
vgg16_155894:	(
vgg16_155896:
vgg16_155898:	(
vgg16_155900:
vgg16_155902:	(
vgg16_155904:
vgg16_155906:	(
vgg16_155908:
vgg16_155910:	(
vgg16_155912:
vgg16_155914:	(
vgg16_155916:
vgg16_155918:	(
vgg16_155920:
vgg16_155922:	(
vgg16_155924:
vgg16_155926:	(
vgg16_155928:
vgg16_155930:	(
vgg16_155932:
vgg16_155934:	
dense_155937:	
dense_155939:
identity¢dense/StatefulPartitionedCall¢vgg16/StatefulPartitionedCallí
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_inputvgg16_155884vgg16_155886vgg16_155888vgg16_155890vgg16_155892vgg16_155894vgg16_155896vgg16_155898vgg16_155900vgg16_155902vgg16_155904vgg16_155906vgg16_155908vgg16_155910vgg16_155912vgg16_155914vgg16_155916vgg16_155918vgg16_155920vgg16_155922vgg16_155924vgg16_155926vgg16_155928vgg16_155930vgg16_155932vgg16_155934*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_155165
dense/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0dense_155937dense_155939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_155498u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
%
_user_specified_namevgg16_input

c
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
m
B__inference_lambda_layer_call_and_return_conditional_losses_158341

inputs
identity¢	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖ_
map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ä
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs%map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
map_while_body_158111*!
condR
map_while_cond_158110*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààR
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2
	map/while	map/while:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs
¡

ó
A__inference_dense_layer_call_and_return_conditional_losses_155498

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
³
&__inference_vgg16_layer_call_fn_155277
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_155165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
Ó
µ
)map_while_random_flip_up_down_true_156495\
Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityv
,map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
'map/while/random_flip_up_down/ReverseV2	ReverseV2Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency5map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
&map/while/random_flip_up_down/IdentityIdentity0map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ì
ð
+__inference_sequential_layer_call_fn_158467

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


H__inference_block4_conv2_layer_call_and_return_conditional_losses_159273

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 


A__inference_model_layer_call_and_return_conditional_losses_156271

inputs+
sequential_156198:@
sequential_156200:@+
sequential_156202:@@
sequential_156204:@,
sequential_156206:@ 
sequential_156208:	-
sequential_156210: 
sequential_156212:	-
sequential_156214: 
sequential_156216:	-
sequential_156218: 
sequential_156220:	-
sequential_156222: 
sequential_156224:	-
sequential_156226: 
sequential_156228:	-
sequential_156230: 
sequential_156232:	-
sequential_156234: 
sequential_156236:	-
sequential_156238: 
sequential_156240:	-
sequential_156242: 
sequential_156244:	-
sequential_156246: 
sequential_156248:	$
sequential_156250:	
sequential_156252:
identity	

identity_1¢lambda/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallÑ
lambda/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156196¾
"sequential/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0sequential_156198sequential_156200sequential_156202sequential_156204sequential_156206sequential_156208sequential_156210sequential_156212sequential_156214sequential_156216sequential_156218sequential_156220sequential_156222sequential_156224sequential_156226sequential_156228sequential_156230sequential_156232sequential_156234sequential_156236sequential_156238sequential_156240sequential_156242sequential_156244sequential_156246sequential_156248sequential_156250sequential_156252*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155505ê
PROBABILITIES/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156259Õ
CLASSES/PartitionedCallPartitionedCall&PROBABILITIES/PartitionedCall:output:0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156267k
IdentityIdentity CLASSES/PartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

Identity_1Identity&PROBABILITIES/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lambda/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
m
B__inference_lambda_layer_call_and_return_conditional_losses_156618

inputs
identity¢	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖ_
map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ä
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs%map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
map_while_body_156388*!
condR
map_while_cond_156387*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààR
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2
	map/while	map/while:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Á
-map_while_random_flip_left_right_false_157926a
]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityÓ
)map/while/random_flip_left_right/IdentityIdentity]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 


A__inference_model_layer_call_and_return_conditional_losses_156940	
bytes+
sequential_156879:@
sequential_156881:@+
sequential_156883:@@
sequential_156885:@,
sequential_156887:@ 
sequential_156889:	-
sequential_156891: 
sequential_156893:	-
sequential_156895: 
sequential_156897:	-
sequential_156899: 
sequential_156901:	-
sequential_156903: 
sequential_156905:	-
sequential_156907: 
sequential_156909:	-
sequential_156911: 
sequential_156913:	-
sequential_156915: 
sequential_156917:	-
sequential_156919: 
sequential_156921:	-
sequential_156923: 
sequential_156925:	-
sequential_156927: 
sequential_156929:	$
sequential_156931:	
sequential_156933:
identity	

identity_1¢lambda/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallÐ
lambda/StatefulPartitionedCallStatefulPartitionedCallbytes*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156196¾
"sequential/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0sequential_156879sequential_156881sequential_156883sequential_156885sequential_156887sequential_156889sequential_156891sequential_156893sequential_156895sequential_156897sequential_156899sequential_156901sequential_156903sequential_156905sequential_156907sequential_156909sequential_156911sequential_156913sequential_156915sequential_156917sequential_156919sequential_156921sequential_156923sequential_156925sequential_156927sequential_156929sequential_156931sequential_156933*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155505ê
PROBABILITIES/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156259Õ
CLASSES/PartitionedCallPartitionedCall&PROBABILITIES/PartitionedCall:output:0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156267k
IdentityIdentity CLASSES/PartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

Identity_1Identity&PROBABILITIES/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lambda/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes

c
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block4_conv1_layer_call_and_return_conditional_losses_159253

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
Á
,map_while_random_flip_left_right_true_158171b
^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityy
/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
*map/while/random_flip_left_right/ReverseV2	ReverseV2^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency8map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
)map/while/random_flip_left_right/IdentityIdentity3map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³
H
,__inference_block5_pool_layer_call_fn_159368

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block1_conv1_layer_call_and_return_conditional_losses_159083

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
¾	
æ
lambda_map_while_cond_1575002
.lambda_map_while_lambda_map_while_loop_counter-
)lambda_map_while_lambda_map_strided_slice 
lambda_map_while_placeholder"
lambda_map_while_placeholder_12
.lambda_map_while_less_lambda_map_strided_sliceJ
Flambda_map_while_lambda_map_while_cond_157500___redundant_placeholder0
lambda_map_while_identity

lambda/map/while/LessLesslambda_map_while_placeholder.lambda_map_while_less_lambda_map_strided_slice*
T0*
_output_shapes
: 
lambda/map/while/Less_1Less.lambda_map_while_lambda_map_while_loop_counter)lambda_map_while_lambda_map_strided_slice*
T0*
_output_shapes
: y
lambda/map/while/LogicalAnd
LogicalAndlambda/map/while/Less_1:z:0lambda/map/while/Less:z:0*
_output_shapes
: g
lambda/map/while/IdentityIdentitylambda/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "?
lambda_map_while_identity"lambda/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


H__inference_block5_conv1_layer_call_and_return_conditional_losses_159323

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_block3_pool_layer_call_fn_159228

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ú
`
'__inference_lambda_layer_call_fn_157844

inputs
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156196y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs


H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ç
"__inference__traced_restore_159646
file_prefix>
$assignvariableop_block1_conv1_kernel:@2
$assignvariableop_1_block1_conv1_bias:@@
&assignvariableop_2_block1_conv2_kernel:@@2
$assignvariableop_3_block1_conv2_bias:@A
&assignvariableop_4_block2_conv1_kernel:@3
$assignvariableop_5_block2_conv1_bias:	B
&assignvariableop_6_block2_conv2_kernel:3
$assignvariableop_7_block2_conv2_bias:	B
&assignvariableop_8_block3_conv1_kernel:3
$assignvariableop_9_block3_conv1_bias:	C
'assignvariableop_10_block3_conv2_kernel:4
%assignvariableop_11_block3_conv2_bias:	C
'assignvariableop_12_block3_conv3_kernel:4
%assignvariableop_13_block3_conv3_bias:	C
'assignvariableop_14_block4_conv1_kernel:4
%assignvariableop_15_block4_conv1_bias:	C
'assignvariableop_16_block4_conv2_kernel:4
%assignvariableop_17_block4_conv2_bias:	C
'assignvariableop_18_block4_conv3_kernel:4
%assignvariableop_19_block4_conv3_bias:	C
'assignvariableop_20_block5_conv1_kernel:4
%assignvariableop_21_block5_conv1_bias:	C
'assignvariableop_22_block5_conv2_kernel:4
%assignvariableop_23_block5_conv2_bias:	C
'assignvariableop_24_block5_conv3_kernel:4
%assignvariableop_25_block5_conv3_bias:	3
 assignvariableop_26_dense_kernel:	,
assignvariableop_27_dense_bias:#
assignvariableop_28_decay: +
!assignvariableop_29_learning_rate: &
assignvariableop_30_momentum: &
assignvariableop_31_sgd_iter:	 #
assignvariableop_32_total: #
assignvariableop_33_count: %
assignvariableop_34_total_1: %
assignvariableop_35_count_1: @
-assignvariableop_36_sgd_dense_kernel_momentum:	9
+assignvariableop_37_sgd_dense_bias_momentum:
identity_39¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*¡
valueB'B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_variables/26/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_variables/27/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¾
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*²
_output_shapes
:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block4_conv1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block4_conv1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block5_conv1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block5_conv1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block5_conv2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block5_conv2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_dense_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_decayIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp!assignvariableop_29_learning_rateIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_sgd_iterIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp-assignvariableop_36_sgd_dense_kernel_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp+assignvariableop_37_sgd_dense_bias_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_39IdentityIdentity_38:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ø
û
&__inference_model_layer_call_fn_157068

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity	

identity_1¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_156271k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
¦
map_while_body_158111$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor¢9map/while/central_crop/assert_greater_equal/Assert/Assert¢@map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Cmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertd
!map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB 
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholder*map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0
map/while/DecodeJpeg
DecodeJpeg4map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
map/while/CastCastmap/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿX
map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
map/while/truedivRealDivmap/while/Cast:y:0map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
&map/while/random_flip_left_right/ShapeShapemap/while/truediv:z:0*
T0*
_output_shapes
:
4map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
.map/while/random_flip_left_right/strided_sliceStridedSlice/map/while/random_flip_left_right/Shape:output:0=map/while/random_flip_left_right/strided_slice/stack:output:0?map/while/random_flip_left_right/strided_slice/stack_1:output:0?map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskx
6map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : è
Amap/while/random_flip_left_right/assert_positive/assert_less/LessLess?map/while/random_flip_left_right/assert_positive/Const:output:07map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Bmap/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ó
@map/while/random_flip_left_right/assert_positive/assert_less/AllAllEmap/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Kmap/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: ´
Imap/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¼
Qmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertImap/while/random_flip_left_right/assert_positive/assert_less/All:output:0Zmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 g
%map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :y
7map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :å
Bmap/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.map/while/random_flip_left_right/Rank:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: |
:map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :»
;map/while/random_flip_left_right/assert_greater_equal/rangeRangeJmap/while/random_flip_left_right/assert_greater_equal/range/start:output:0Cmap/while/random_flip_left_right/assert_greater_equal/Rank:output:0Jmap/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: æ
9map/while/random_flip_left_right/assert_greater_equal/AllAllFmap/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dmap/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: ®
Bmap/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.°
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Å
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = ¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¹
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Ë
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = Ë
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertBmap/while/random_flip_left_right/assert_greater_equal/All:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.map/while/random_flip_left_right/Rank:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0K^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Î
3map/while/random_flip_left_right/control_dependencyIdentitymap/while/truediv:z:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*$
_class
loc:@map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
5map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB x
3map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
=map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniform>map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0á
3map/while/random_flip_left_right/random_uniform/MulMulFmap/while/random_flip_left_right/random_uniform/RandomUniform:output:0<map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: l
'map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
%map/while/random_flip_left_right/LessLess7map/while/random_flip_left_right/random_uniform/Mul:z:00map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ó
 map/while/random_flip_left_rightStatelessIf)map/while/random_flip_left_right/Less:z:0<map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-map_while_random_flip_left_right_false_158172*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,map_while_random_flip_left_right_true_158171
)map/while/random_flip_left_right/IdentityIdentity)map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
#map/while/random_flip_up_down/ShapeShape2map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
1map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
3map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
+map/while/random_flip_up_down/strided_sliceStridedSlice,map/while/random_flip_up_down/Shape:output:0:map/while/random_flip_up_down/strided_slice/stack:output:0<map/while/random_flip_up_down/strided_slice/stack_1:output:0<map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masku
3map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ß
>map/while/random_flip_up_down/assert_positive/assert_less/LessLess<map/while/random_flip_up_down/assert_positive/Const:output:04map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
?map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ê
=map/while/random_flip_up_down/assert_positive/assert_less/AllAllBmap/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Hmap/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ±
Fmap/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¹
Nmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ú
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFmap/while/random_flip_up_down/assert_positive/assert_less/All:output:0Wmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 d
"map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :v
4map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ü
?map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+map/while/random_flip_up_down/Rank:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: y
7map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¯
8map/while/random_flip_up_down/assert_greater_equal/rangeRangeGmap/while/random_flip_up_down/assert_greater_equal/range/start:output:0@map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Gmap/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: Ý
6map/while/random_flip_up_down/assert_greater_equal/AllAllCmap/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Amap/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: «
?map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = ¿
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = ³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = Å
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = °
@map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssert?map/while/random_flip_up_down/assert_greater_equal/All:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+map/while/random_flip_up_down/Rank:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0H^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ú
0map/while/random_flip_up_down/control_dependencyIdentity2map/while/random_flip_left_right/Identity:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
2map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB u
0map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
:map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniform;map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0Ø
0map/while/random_flip_up_down/random_uniform/MulMulCmap/while/random_flip_up_down/random_uniform/RandomUniform:output:09map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: i
$map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
"map/while/random_flip_up_down/LessLess4map/while/random_flip_up_down/random_uniform/Mul:z:0-map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: ä
map/while/random_flip_up_downStatelessIf&map/while/random_flip_up_down/Less:z:09map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*map_while_random_flip_up_down_false_158219*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)map_while_random_flip_up_down_true_158218
&map/while/random_flip_up_down/IdentityIdentity&map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
(map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½k
&map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Dmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      Ý
?map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterMmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
?map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21map/while/stateless_random_uniform/shape:output:0Emap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Imap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hmap/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&map/while/stateless_random_uniform/subSub/map/while/stateless_random_uniform/max:output:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&map/while/stateless_random_uniform/mulMulDmap/while/stateless_random_uniform/StatelessRandomUniformV2:output:0*map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"map/while/stateless_random_uniformAddV2*map/while/stateless_random_uniform/mul:z:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
map/while/adjust_brightnessAddV2/map/while/random_flip_up_down/Identity:output:0&map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$map/while/adjust_brightness/IdentityIdentitymap/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
&map/while/random_uniform/RandomUniformRandomUniform'map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0
map/while/random_uniform/subSub%map/while/random_uniform/max:output:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: 
map/while/random_uniform/mulMul/map/while/random_uniform/RandomUniform:output:0 map/while/random_uniform/sub:z:0*
T0*
_output_shapes
: 
map/while/random_uniformAddV2 map/while/random_uniform/mul:z:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ã
,map/while/adjust_saturation/AdjustSaturationAdjustSaturation-map/while/adjust_brightness/Identity:output:0map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
$map/while/adjust_saturation/IdentityIdentity5map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
map/while/central_crop/ShapeShape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
*map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
,map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$map/while/central_crop/strided_sliceStridedSlice%map/while/central_crop/Shape:output:03map/while/central_crop/strided_slice/stack:output:05map/while/central_crop/strided_slice/stack_1:output:05map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
,map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ê
7map/while/central_crop/assert_positive/assert_less/LessLess5map/while/central_crop/assert_positive/Const:output:0-map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
8map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: Õ
6map/while/central_crop/assert_positive/assert_less/AllAll;map/while/central_crop/assert_positive/assert_less/Less:z:0Amap/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ª
?map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Gmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Â
@map/while/central_crop/assert_positive/assert_less/Assert/AssertAssert?map/while/central_crop/assert_positive/assert_less/All:output:0Pmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 ]
map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ç
8map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual$map/while/central_crop/Rank:output:06map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: r
0map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
1map/while/central_crop/assert_greater_equal/rangeRange@map/while/central_crop/assert_greater_equal/range/start:output:09map/while/central_crop/assert_greater_equal/Rank:output:0@map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: È
/map/while/central_crop/assert_greater_equal/AllAll<map/while/central_crop/assert_greater_equal/GreaterEqual:z:0:map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: ¤
8map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¦
:map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
:map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ±
:map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¥
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ·
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ñ
9map/while/central_crop/assert_greater_equal/Assert/AssertAssert8map/while/central_crop/assert_greater_equal/All:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:0$map/while/central_crop/Rank:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:06map/while/central_crop/assert_greater_equal/y:output:0A^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Û
)map/while/central_crop/control_dependencyIdentity-map/while/adjust_saturation/Identity:output:0:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*7
_class-
+)loc:@map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
map/while/central_crop/Shape_1Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_1StridedSlice'map/while/central_crop/Shape_1:output:05map/while/central_crop/strided_slice_1/stack:output:07map/while/central_crop/strided_slice_1/stack_1:output:07map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
map/while/central_crop/Shape_2Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_2StridedSlice'map/while/central_crop/Shape_2:output:05map/while/central_crop/strided_slice_2/stack:output:07map/while/central_crop/strided_slice_2/stack_1:output:07map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
map/while/central_crop/CastCast/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *;]X?
map/while/central_crop/Cast_1Cast(map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mulMulmap/while/central_crop/Cast:y:0!map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: 
map/while/central_crop/subSubmap/while/central_crop/Cast:y:0map/while/central_crop/mul:z:0*
T0*
_output_shapes
: i
 map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
map/while/central_crop/truedivRealDivmap/while/central_crop/sub:z:0)map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: y
map/while/central_crop/Cast_2Cast"map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/Cast_3Cast/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *;]X?
map/while/central_crop/Cast_4Cast(map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mul_1Mul!map/while/central_crop/Cast_3:y:0!map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_1Sub!map/while/central_crop/Cast_3:y:0 map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: k
"map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
 map/while/central_crop/truediv_1RealDiv map/while/central_crop/sub_1:z:0+map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: {
map/while/central_crop/Cast_5Cast$map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: `
map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_2Mul!map/while/central_crop/Cast_2:y:0'map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_2Sub/map/while/central_crop/strided_slice_1:output:0 map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: `
map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_3Mul!map/while/central_crop/Cast_5:y:0'map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_3Sub/map/while/central_crop/strided_slice_2:output:0 map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: `
map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Á
map/while/central_crop/stackPack!map/while/central_crop/Cast_2:y:0!map/while/central_crop/Cast_5:y:0'map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:k
 map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
map/while/central_crop/stack_1Pack map/while/central_crop/sub_2:z:0 map/while/central_crop/sub_3:z:0)map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:ð
map/while/central_crop/SliceSlice-map/while/adjust_saturation/Identity:output:0%map/while/central_crop/stack:output:0'map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
map/while/resize/ExpandDims
ExpandDims%map/while/central_crop/Slice:output:0(map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ä
map/while/resize/ResizeBilinearResizeBilinear$map/while/resize/ExpandDims:output:0map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(
map/while/resize/SqueezeSqueeze0map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 Ö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder!map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒï
map/while/NoOpNoOp:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/AssertD^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertA^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2v
9map/while/central_crop/assert_greater_equal/Assert/Assert9map/while/central_crop/assert_greater_equal/Assert/Assert2
@map/while/central_crop/assert_positive/assert_less/Assert/Assert@map/while/central_crop/assert_positive/assert_less/Assert/Assert2
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertCmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert2
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertJmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertGmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Á
-map_while_random_flip_left_right_false_158172a
]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityÓ
)map/while/random_flip_left_right/IdentityIdentity]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv2_layer_call_fn_159192

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ëÔ
ä
A__inference_model_layer_call_and_return_conditional_losses_157774

inputsV
<sequential_vgg16_block1_conv1_conv2d_readvariableop_resource:@K
=sequential_vgg16_block1_conv1_biasadd_readvariableop_resource:@V
<sequential_vgg16_block1_conv2_conv2d_readvariableop_resource:@@K
=sequential_vgg16_block1_conv2_biasadd_readvariableop_resource:@W
<sequential_vgg16_block2_conv1_conv2d_readvariableop_resource:@L
=sequential_vgg16_block2_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block2_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block2_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv3_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv3_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv3_biasadd_readvariableop_resource:	B
/sequential_dense_matmul_readvariableop_resource:	>
0sequential_dense_biasadd_readvariableop_resource:
identity	

identity_1¢lambda/map/while¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpF
lambda/map/ShapeShapeinputs*
T0*
_output_shapes
:h
lambda/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lambda/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lambda/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lambda/map/strided_sliceStridedSlicelambda/map/Shape:output:0'lambda/map/strided_slice/stack:output:0)lambda/map/strided_slice/stack_1:output:0)lambda/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
&lambda/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
lambda/map/TensorArrayV2TensorListReserve/lambda/map/TensorArrayV2/element_shape:output:0!lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖf
#lambda/map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ò
2lambda/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs,lambda/map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖR
lambda/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : s
(lambda/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
lambda/map/TensorArrayV2_1TensorListReserve1lambda/map/TensorArrayV2_1/element_shape:output:0!lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
lambda/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
lambda/map/whileWhile&lambda/map/while/loop_counter:output:0!lambda/map/strided_slice:output:0lambda/map/Const:output:0#lambda/map/TensorArrayV2_1:handle:0!lambda/map/strided_slice:output:0Blambda/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *(
body R
lambda_map_while_body_157501*(
cond R
lambda_map_while_cond_157500*
output_shapes
: : : : : : 
;lambda/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      é
-lambda/map/TensorArrayV2Stack/TensorListStackTensorListStacklambda/map/while:output:3Dlambda/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0¸
3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
$sequential/vgg16/block1_conv1/Conv2DConv2D6lambda/map/TensorArrayV2Stack/TensorListStack:tensor:0;sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
®
4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%sequential/vgg16/block1_conv1/BiasAddBiasAdd-sequential/vgg16/block1_conv1/Conv2D:output:0<sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"sequential/vgg16/block1_conv1/ReluRelu.sequential/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
$sequential/vgg16/block1_conv2/Conv2DConv2D0sequential/vgg16/block1_conv1/Relu:activations:0;sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
®
4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%sequential/vgg16/block1_conv2/BiasAddBiasAdd-sequential/vgg16/block1_conv2/Conv2D:output:0<sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"sequential/vgg16/block1_conv2/ReluRelu.sequential/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Î
$sequential/vgg16/block1_pool/MaxPoolMaxPool0sequential/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
¹
3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ý
$sequential/vgg16/block2_conv1/Conv2DConv2D-sequential/vgg16/block1_pool/MaxPool:output:0;sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¯
4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block2_conv1/BiasAddBiasAdd-sequential/vgg16/block2_conv1/Conv2D:output:0<sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"sequential/vgg16/block2_conv1/ReluRelu.sequential/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppº
3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block2_conv2/Conv2DConv2D0sequential/vgg16/block2_conv1/Relu:activations:0;sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¯
4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block2_conv2/BiasAddBiasAdd-sequential/vgg16/block2_conv2/Conv2D:output:0<sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"sequential/vgg16/block2_conv2/ReluRelu.sequential/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÏ
$sequential/vgg16/block2_pool/MaxPoolMaxPool0sequential/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block3_conv1/Conv2DConv2D-sequential/vgg16/block2_pool/MaxPool:output:0;sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv1/BiasAddBiasAdd-sequential/vgg16/block3_conv1/Conv2D:output:0<sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv1/ReluRelu.sequential/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88º
3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block3_conv2/Conv2DConv2D0sequential/vgg16/block3_conv1/Relu:activations:0;sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv2/BiasAddBiasAdd-sequential/vgg16/block3_conv2/Conv2D:output:0<sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv2/ReluRelu.sequential/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88º
3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block3_conv3/Conv2DConv2D0sequential/vgg16/block3_conv2/Relu:activations:0;sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv3/BiasAddBiasAdd-sequential/vgg16/block3_conv3/Conv2D:output:0<sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv3/ReluRelu.sequential/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ï
$sequential/vgg16/block3_pool/MaxPoolMaxPool0sequential/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block4_conv1/Conv2DConv2D-sequential/vgg16/block3_pool/MaxPool:output:0;sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv1/BiasAddBiasAdd-sequential/vgg16/block4_conv1/Conv2D:output:0<sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv1/ReluRelu.sequential/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block4_conv2/Conv2DConv2D0sequential/vgg16/block4_conv1/Relu:activations:0;sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv2/BiasAddBiasAdd-sequential/vgg16/block4_conv2/Conv2D:output:0<sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv2/ReluRelu.sequential/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block4_conv3/Conv2DConv2D0sequential/vgg16/block4_conv2/Relu:activations:0;sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv3/BiasAddBiasAdd-sequential/vgg16/block4_conv3/Conv2D:output:0<sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv3/ReluRelu.sequential/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$sequential/vgg16/block4_pool/MaxPoolMaxPool0sequential/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block5_conv1/Conv2DConv2D-sequential/vgg16/block4_pool/MaxPool:output:0;sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv1/BiasAddBiasAdd-sequential/vgg16/block5_conv1/Conv2D:output:0<sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv1/ReluRelu.sequential/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block5_conv2/Conv2DConv2D0sequential/vgg16/block5_conv1/Relu:activations:0;sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv2/BiasAddBiasAdd-sequential/vgg16/block5_conv2/Conv2D:output:0<sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv2/ReluRelu.sequential/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block5_conv3/Conv2DConv2D0sequential/vgg16/block5_conv2/Relu:activations:0;sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv3/BiasAddBiasAdd-sequential/vgg16/block5_conv3/Conv2D:output:0<sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv3/ReluRelu.sequential/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$sequential/vgg16/block5_pool/MaxPoolMaxPool0sequential/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

@sequential/vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ã
.sequential/vgg16/global_average_pooling2d/MeanMean-sequential/vgg16/block5_pool/MaxPool:output:0Isequential/vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¼
sequential/dense/MatMulMatMul7sequential/vgg16/global_average_pooling2d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
CLASSES/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
CLASSES/ArgMaxArgMax"sequential/dense/Softmax:softmax:0!CLASSES/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityCLASSES/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_1Identity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp^lambda/map/while(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp5^sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
lambda/map/whilelambda/map/while2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2l
4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv2_layer_call_fn_159262

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block5_pool_layer_call_and_return_conditional_losses_159373

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
H
,__inference_block1_pool_layer_call_fn_159108

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block3_conv1_layer_call_fn_159172

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv2_layer_call_fn_159332

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
³
&__inference_vgg16_layer_call_fn_154901
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_154846p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1

õ
:model_lambda_map_while_random_flip_left_right_false_154261{
wmodel_lambda_map_while_random_flip_left_right_identity_model_lambda_map_while_random_flip_left_right_control_dependency:
6model_lambda_map_while_random_flip_left_right_identityú
6model/lambda/map/while/random_flip_left_right/IdentityIdentitywmodel_lambda_map_while_random_flip_left_right_identity_model_lambda_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"y
6model_lambda_map_while_random_flip_left_right_identity?model/lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ú
Á
,map_while_random_flip_left_right_true_156448b
^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityy
/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
*map/while/random_flip_left_right/ReverseV2	ReverseV2^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency8map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
)map/while/random_flip_left_right/IdentityIdentity3map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
è
ö
A__inference_vgg16_layer_call_and_return_conditional_losses_158941

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      °
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ë
²
&__inference_vgg16_layer_call_fn_158782

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_154846p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ü
m
B__inference_lambda_layer_call_and_return_conditional_losses_156196

inputs
identity¢	map/while?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖ_
map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ä
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs%map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ñ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *!
bodyR
map_while_body_155966*!
condR
map_while_cond_155965*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààR
NoOpNoOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2
	map/while	map/while:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block5_conv2_layer_call_and_return_conditional_losses_159343

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
õ
+__inference_sequential_layer_call_fn_155819
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
%
_user_specified_namevgg16_input
 


A__inference_model_layer_call_and_return_conditional_losses_157005	
bytes+
sequential_156944:@
sequential_156946:@+
sequential_156948:@@
sequential_156950:@,
sequential_156952:@ 
sequential_156954:	-
sequential_156956: 
sequential_156958:	-
sequential_156960: 
sequential_156962:	-
sequential_156964: 
sequential_156966:	-
sequential_156968: 
sequential_156970:	-
sequential_156972: 
sequential_156974:	-
sequential_156976: 
sequential_156978:	-
sequential_156980: 
sequential_156982:	-
sequential_156984: 
sequential_156986:	-
sequential_156988: 
sequential_156990:	-
sequential_156992: 
sequential_156994:	$
sequential_156996:	
sequential_156998:
identity	

identity_1¢lambda/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallÐ
lambda/StatefulPartitionedCallStatefulPartitionedCallbytes*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156618¾
"sequential/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0sequential_156944sequential_156946sequential_156948sequential_156950sequential_156952sequential_156954sequential_156956sequential_156958sequential_156960sequential_156962sequential_156964sequential_156966sequential_156968sequential_156970sequential_156972sequential_156974sequential_156976sequential_156978sequential_156980sequential_156982sequential_156984sequential_156986sequential_156988sequential_156990sequential_156992sequential_156994sequential_156996sequential_156998*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155699ê
PROBABILITIES/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156361Õ
CLASSES/PartitionedCallPartitionedCall&PROBABILITIES/PartitionedCall:output:0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156346k
IdentityIdentity CLASSES/PartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

Identity_1Identity&PROBABILITIES/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lambda/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes
¤	
õ
9model_lambda_map_while_random_flip_left_right_true_154260|
xmodel_lambda_map_while_random_flip_left_right_reversev2_model_lambda_map_while_random_flip_left_right_control_dependency:
6model_lambda_map_while_random_flip_left_right_identity
<model/lambda/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Ä
7model/lambda/map/while/random_flip_left_right/ReverseV2	ReverseV2xmodel_lambda_map_while_random_flip_left_right_reversev2_model_lambda_map_while_random_flip_left_right_control_dependencyEmodel/lambda/map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
6model/lambda/map/while/random_flip_left_right/IdentityIdentity@model/lambda/map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"y
6model_lambda_map_while_random_flip_left_right_identity?model/lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø
Þ
F__inference_sequential_layer_call_and_return_conditional_losses_158685

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	M
1vgg16_block2_conv2_conv2d_readvariableop_resource:A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv1_conv2d_readvariableop_resource:A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	M
1vgg16_block3_conv2_conv2d_readvariableop_resource:A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	M
1vgg16_block3_conv3_conv2d_readvariableop_resource:A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	M
1vgg16_block4_conv1_conv2d_readvariableop_resource:A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	M
1vgg16_block4_conv2_conv2d_readvariableop_resource:A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	M
1vgg16_block4_conv3_conv2d_readvariableop_resource:A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	M
1vgg16_block5_conv1_conv2d_readvariableop_resource:A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	M
1vgg16_block5_conv2_conv2d_readvariableop_resource:A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	M
1vgg16_block5_conv3_conv2d_readvariableop_resource:A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	7
$dense_matmul_readvariableop_resource:	3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢)vgg16/block1_conv1/BiasAdd/ReadVariableOp¢(vgg16/block1_conv1/Conv2D/ReadVariableOp¢)vgg16/block1_conv2/BiasAdd/ReadVariableOp¢(vgg16/block1_conv2/Conv2D/ReadVariableOp¢)vgg16/block2_conv1/BiasAdd/ReadVariableOp¢(vgg16/block2_conv1/Conv2D/ReadVariableOp¢)vgg16/block2_conv2/BiasAdd/ReadVariableOp¢(vgg16/block2_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv1/BiasAdd/ReadVariableOp¢(vgg16/block3_conv1/Conv2D/ReadVariableOp¢)vgg16/block3_conv2/BiasAdd/ReadVariableOp¢(vgg16/block3_conv2/Conv2D/ReadVariableOp¢)vgg16/block3_conv3/BiasAdd/ReadVariableOp¢(vgg16/block3_conv3/Conv2D/ReadVariableOp¢)vgg16/block4_conv1/BiasAdd/ReadVariableOp¢(vgg16/block4_conv1/Conv2D/ReadVariableOp¢)vgg16/block4_conv2/BiasAdd/ReadVariableOp¢(vgg16/block4_conv2/Conv2D/ReadVariableOp¢)vgg16/block4_conv3/BiasAdd/ReadVariableOp¢(vgg16/block4_conv3/Conv2D/ReadVariableOp¢)vgg16/block5_conv1/BiasAdd/ReadVariableOp¢(vgg16/block5_conv1/Conv2D/ReadVariableOp¢)vgg16/block5_conv2/BiasAdd/ReadVariableOp¢(vgg16/block5_conv2/Conv2D/ReadVariableOp¢)vgg16/block5_conv3/BiasAdd/ReadVariableOp¢(vgg16/block5_conv3/Conv2D/ReadVariableOp¢
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
£
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¹
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¤
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¹
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

5vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Â
#vgg16/global_average_pooling2d/MeanMean"vgg16/block5_pool/MaxPool:output:0>vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense/MatMulMatMul,vgg16/global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
û
µ
*map_while_random_flip_up_down_false_157973[
Wmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityÊ
&map/while/random_flip_up_down/IdentityIdentityWmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ü
ö
F__inference_sequential_layer_call_and_return_conditional_losses_155505

inputs&
vgg16_155434:@
vgg16_155436:@&
vgg16_155438:@@
vgg16_155440:@'
vgg16_155442:@
vgg16_155444:	(
vgg16_155446:
vgg16_155448:	(
vgg16_155450:
vgg16_155452:	(
vgg16_155454:
vgg16_155456:	(
vgg16_155458:
vgg16_155460:	(
vgg16_155462:
vgg16_155464:	(
vgg16_155466:
vgg16_155468:	(
vgg16_155470:
vgg16_155472:	(
vgg16_155474:
vgg16_155476:	(
vgg16_155478:
vgg16_155480:	(
vgg16_155482:
vgg16_155484:	
dense_155499:	
dense_155501:
identity¢dense/StatefulPartitionedCall¢vgg16/StatefulPartitionedCallè
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_155434vgg16_155436vgg16_155438vgg16_155440vgg16_155442vgg16_155444vgg16_155446vgg16_155448vgg16_155450vgg16_155452vgg16_155454vgg16_155456vgg16_155458vgg16_155460vgg16_155462vgg16_155464vgg16_155466vgg16_155468vgg16_155470vgg16_155472vgg16_155474vgg16_155476vgg16_155478vgg16_155480vgg16_155482vgg16_155484*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_154846
dense/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0dense_155499dense_155501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_155498u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
­
¦
map_while_body_156388$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor¢9map/while/central_crop/assert_greater_equal/Assert/Assert¢@map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Cmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertd
!map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB 
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholder*map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0
map/while/DecodeJpeg
DecodeJpeg4map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
map/while/CastCastmap/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿX
map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
map/while/truedivRealDivmap/while/Cast:y:0map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
&map/while/random_flip_left_right/ShapeShapemap/while/truediv:z:0*
T0*
_output_shapes
:
4map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
.map/while/random_flip_left_right/strided_sliceStridedSlice/map/while/random_flip_left_right/Shape:output:0=map/while/random_flip_left_right/strided_slice/stack:output:0?map/while/random_flip_left_right/strided_slice/stack_1:output:0?map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskx
6map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : è
Amap/while/random_flip_left_right/assert_positive/assert_less/LessLess?map/while/random_flip_left_right/assert_positive/Const:output:07map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Bmap/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ó
@map/while/random_flip_left_right/assert_positive/assert_less/AllAllEmap/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Kmap/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: ´
Imap/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¼
Qmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertImap/while/random_flip_left_right/assert_positive/assert_less/All:output:0Zmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 g
%map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :y
7map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :å
Bmap/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.map/while/random_flip_left_right/Rank:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: |
:map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :»
;map/while/random_flip_left_right/assert_greater_equal/rangeRangeJmap/while/random_flip_left_right/assert_greater_equal/range/start:output:0Cmap/while/random_flip_left_right/assert_greater_equal/Rank:output:0Jmap/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: æ
9map/while/random_flip_left_right/assert_greater_equal/AllAllFmap/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dmap/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: ®
Bmap/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.°
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Å
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = ¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¹
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Ë
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = Ë
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertBmap/while/random_flip_left_right/assert_greater_equal/All:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.map/while/random_flip_left_right/Rank:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0K^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Î
3map/while/random_flip_left_right/control_dependencyIdentitymap/while/truediv:z:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*$
_class
loc:@map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
5map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB x
3map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
=map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniform>map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0á
3map/while/random_flip_left_right/random_uniform/MulMulFmap/while/random_flip_left_right/random_uniform/RandomUniform:output:0<map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: l
'map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
%map/while/random_flip_left_right/LessLess7map/while/random_flip_left_right/random_uniform/Mul:z:00map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ó
 map/while/random_flip_left_rightStatelessIf)map/while/random_flip_left_right/Less:z:0<map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-map_while_random_flip_left_right_false_156449*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,map_while_random_flip_left_right_true_156448
)map/while/random_flip_left_right/IdentityIdentity)map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
#map/while/random_flip_up_down/ShapeShape2map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
1map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
3map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
+map/while/random_flip_up_down/strided_sliceStridedSlice,map/while/random_flip_up_down/Shape:output:0:map/while/random_flip_up_down/strided_slice/stack:output:0<map/while/random_flip_up_down/strided_slice/stack_1:output:0<map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masku
3map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ß
>map/while/random_flip_up_down/assert_positive/assert_less/LessLess<map/while/random_flip_up_down/assert_positive/Const:output:04map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
?map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ê
=map/while/random_flip_up_down/assert_positive/assert_less/AllAllBmap/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Hmap/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ±
Fmap/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¹
Nmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ú
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFmap/while/random_flip_up_down/assert_positive/assert_less/All:output:0Wmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 d
"map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :v
4map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ü
?map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+map/while/random_flip_up_down/Rank:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: y
7map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¯
8map/while/random_flip_up_down/assert_greater_equal/rangeRangeGmap/while/random_flip_up_down/assert_greater_equal/range/start:output:0@map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Gmap/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: Ý
6map/while/random_flip_up_down/assert_greater_equal/AllAllCmap/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Amap/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: «
?map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = ¿
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = ³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = Å
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = °
@map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssert?map/while/random_flip_up_down/assert_greater_equal/All:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+map/while/random_flip_up_down/Rank:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0H^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ú
0map/while/random_flip_up_down/control_dependencyIdentity2map/while/random_flip_left_right/Identity:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
2map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB u
0map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
:map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniform;map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0Ø
0map/while/random_flip_up_down/random_uniform/MulMulCmap/while/random_flip_up_down/random_uniform/RandomUniform:output:09map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: i
$map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
"map/while/random_flip_up_down/LessLess4map/while/random_flip_up_down/random_uniform/Mul:z:0-map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: ä
map/while/random_flip_up_downStatelessIf&map/while/random_flip_up_down/Less:z:09map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*map_while_random_flip_up_down_false_156496*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)map_while_random_flip_up_down_true_156495
&map/while/random_flip_up_down/IdentityIdentity&map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
(map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½k
&map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Dmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      Ý
?map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterMmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
?map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21map/while/stateless_random_uniform/shape:output:0Emap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Imap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hmap/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&map/while/stateless_random_uniform/subSub/map/while/stateless_random_uniform/max:output:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&map/while/stateless_random_uniform/mulMulDmap/while/stateless_random_uniform/StatelessRandomUniformV2:output:0*map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"map/while/stateless_random_uniformAddV2*map/while/stateless_random_uniform/mul:z:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
map/while/adjust_brightnessAddV2/map/while/random_flip_up_down/Identity:output:0&map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$map/while/adjust_brightness/IdentityIdentitymap/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
&map/while/random_uniform/RandomUniformRandomUniform'map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0
map/while/random_uniform/subSub%map/while/random_uniform/max:output:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: 
map/while/random_uniform/mulMul/map/while/random_uniform/RandomUniform:output:0 map/while/random_uniform/sub:z:0*
T0*
_output_shapes
: 
map/while/random_uniformAddV2 map/while/random_uniform/mul:z:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ã
,map/while/adjust_saturation/AdjustSaturationAdjustSaturation-map/while/adjust_brightness/Identity:output:0map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
$map/while/adjust_saturation/IdentityIdentity5map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
map/while/central_crop/ShapeShape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
*map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
,map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$map/while/central_crop/strided_sliceStridedSlice%map/while/central_crop/Shape:output:03map/while/central_crop/strided_slice/stack:output:05map/while/central_crop/strided_slice/stack_1:output:05map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
,map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ê
7map/while/central_crop/assert_positive/assert_less/LessLess5map/while/central_crop/assert_positive/Const:output:0-map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
8map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: Õ
6map/while/central_crop/assert_positive/assert_less/AllAll;map/while/central_crop/assert_positive/assert_less/Less:z:0Amap/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ª
?map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Gmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Â
@map/while/central_crop/assert_positive/assert_less/Assert/AssertAssert?map/while/central_crop/assert_positive/assert_less/All:output:0Pmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 ]
map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ç
8map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual$map/while/central_crop/Rank:output:06map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: r
0map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
1map/while/central_crop/assert_greater_equal/rangeRange@map/while/central_crop/assert_greater_equal/range/start:output:09map/while/central_crop/assert_greater_equal/Rank:output:0@map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: È
/map/while/central_crop/assert_greater_equal/AllAll<map/while/central_crop/assert_greater_equal/GreaterEqual:z:0:map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: ¤
8map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¦
:map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
:map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ±
:map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¥
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ·
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ñ
9map/while/central_crop/assert_greater_equal/Assert/AssertAssert8map/while/central_crop/assert_greater_equal/All:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:0$map/while/central_crop/Rank:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:06map/while/central_crop/assert_greater_equal/y:output:0A^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Û
)map/while/central_crop/control_dependencyIdentity-map/while/adjust_saturation/Identity:output:0:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*7
_class-
+)loc:@map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
map/while/central_crop/Shape_1Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_1StridedSlice'map/while/central_crop/Shape_1:output:05map/while/central_crop/strided_slice_1/stack:output:07map/while/central_crop/strided_slice_1/stack_1:output:07map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
map/while/central_crop/Shape_2Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_2StridedSlice'map/while/central_crop/Shape_2:output:05map/while/central_crop/strided_slice_2/stack:output:07map/while/central_crop/strided_slice_2/stack_1:output:07map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
map/while/central_crop/CastCast/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *³]:?
map/while/central_crop/Cast_1Cast(map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mulMulmap/while/central_crop/Cast:y:0!map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: 
map/while/central_crop/subSubmap/while/central_crop/Cast:y:0map/while/central_crop/mul:z:0*
T0*
_output_shapes
: i
 map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
map/while/central_crop/truedivRealDivmap/while/central_crop/sub:z:0)map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: y
map/while/central_crop/Cast_2Cast"map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/Cast_3Cast/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *³]:?
map/while/central_crop/Cast_4Cast(map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mul_1Mul!map/while/central_crop/Cast_3:y:0!map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_1Sub!map/while/central_crop/Cast_3:y:0 map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: k
"map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
 map/while/central_crop/truediv_1RealDiv map/while/central_crop/sub_1:z:0+map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: {
map/while/central_crop/Cast_5Cast$map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: `
map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_2Mul!map/while/central_crop/Cast_2:y:0'map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_2Sub/map/while/central_crop/strided_slice_1:output:0 map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: `
map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_3Mul!map/while/central_crop/Cast_5:y:0'map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_3Sub/map/while/central_crop/strided_slice_2:output:0 map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: `
map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Á
map/while/central_crop/stackPack!map/while/central_crop/Cast_2:y:0!map/while/central_crop/Cast_5:y:0'map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:k
 map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
map/while/central_crop/stack_1Pack map/while/central_crop/sub_2:z:0 map/while/central_crop/sub_3:z:0)map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:ð
map/while/central_crop/SliceSlice-map/while/adjust_saturation/Identity:output:0%map/while/central_crop/stack:output:0'map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
map/while/resize/ExpandDims
ExpandDims%map/while/central_crop/Slice:output:0(map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ä
map/while/resize/ResizeBilinearResizeBilinear$map/while/resize/ExpandDims:output:0map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(
map/while/resize/SqueezeSqueeze0map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 Ö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder!map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒï
map/while/NoOpNoOp:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/AssertD^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertA^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2v
9map/while/central_crop/assert_greater_equal/Assert/Assert9map/while/central_crop/assert_greater_equal/Assert/Assert2
@map/while/central_crop/assert_positive/assert_less/Assert/Assert@map/while/central_crop/assert_positive/assert_less/Assert/Assert2
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertCmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert2
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertJmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertGmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ø
û
&__inference_model_layer_call_fn_157131

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity	

identity_1¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_156751k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
é
7model_lambda_map_while_random_flip_up_down_false_154308u
qmodel_lambda_map_while_random_flip_up_down_identity_model_lambda_map_while_random_flip_up_down_control_dependency7
3model_lambda_map_while_random_flip_up_down_identityñ
3model/lambda/map/while/random_flip_up_down/IdentityIdentityqmodel_lambda_map_while_random_flip_up_down_identity_model_lambda_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"s
3model_lambda_map_while_random_flip_up_down_identity<model/lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó
µ
)map_while_random_flip_up_down_true_157972\
Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityv
,map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
'map/while/random_flip_up_down/ReverseV2	ReverseV2Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency5map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
&map/while/random_flip_up_down/IdentityIdentity0map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

c
G__inference_block3_pool_layer_call_and_return_conditional_losses_159233

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ñ
0lambda_map_while_random_flip_up_down_true_157608j
flambda_map_while_random_flip_up_down_reversev2_lambda_map_while_random_flip_up_down_control_dependency1
-lambda_map_while_random_flip_up_down_identity}
3lambda/map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:  
.lambda/map/while/random_flip_up_down/ReverseV2	ReverseV2flambda_map_while_random_flip_up_down_reversev2_lambda_map_while_random_flip_up_down_control_dependency<lambda/map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
-lambda/map/while/random_flip_up_down/IdentityIdentity7lambda/map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"g
-lambda_map_while_random_flip_up_down_identity6lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

c
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs
ù
e
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156361

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_CLASSES_layer_call_fn_158713

inputs
identity	­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156346\
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
_
C__inference_CLASSES_layer_call_and_return_conditional_losses_156267

inputs
identity	R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :a
ArgMaxArgMaxinputsArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentityArgMax:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
J
.__inference_PROBABILITIES_layer_call_fn_158695

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156361`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
Ñ
1lambda_map_while_random_flip_up_down_false_157609i
elambda_map_while_random_flip_up_down_identity_lambda_map_while_random_flip_up_down_control_dependency1
-lambda_map_while_random_flip_up_down_identityß
-lambda/map/while/random_flip_up_down/IdentityIdentityelambda_map_while_random_flip_up_down_identity_lambda_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"g
-lambda_map_while_random_flip_up_down_identity6lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Õ
ú
&__inference_model_layer_call_fn_156332	
bytes!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity	

identity_1¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallbytesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_156271k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes
ç

®
"model_lambda_map_while_cond_154199>
:model_lambda_map_while_model_lambda_map_while_loop_counter9
5model_lambda_map_while_model_lambda_map_strided_slice&
"model_lambda_map_while_placeholder(
$model_lambda_map_while_placeholder_1>
:model_lambda_map_while_less_model_lambda_map_strided_sliceV
Rmodel_lambda_map_while_model_lambda_map_while_cond_154199___redundant_placeholder0#
model_lambda_map_while_identity
¤
model/lambda/map/while/LessLess"model_lambda_map_while_placeholder:model_lambda_map_while_less_model_lambda_map_strided_slice*
T0*
_output_shapes
: ¹
model/lambda/map/while/Less_1Less:model_lambda_map_while_model_lambda_map_while_loop_counter5model_lambda_map_while_model_lambda_map_strided_slice*
T0*
_output_shapes
: 
!model/lambda/map/while/LogicalAnd
LogicalAnd!model/lambda/map/while/Less_1:z:0model/lambda/map/while/Less:z:0*
_output_shapes
: s
model/lambda/map/while/IdentityIdentity%model/lambda/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "K
model_lambda_map_while_identity(model/lambda/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ú
`
'__inference_lambda_layer_call_fn_157849

inputs
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156618y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àÐ
È
lambda_map_while_body_1575012
.lambda_map_while_lambda_map_while_loop_counter-
)lambda_map_while_lambda_map_strided_slice 
lambda_map_while_placeholder"
lambda_map_while_placeholder_11
-lambda_map_while_lambda_map_strided_slice_1_0m
ilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0
lambda_map_while_identity
lambda_map_while_identity_1
lambda_map_while_identity_2
lambda_map_while_identity_3/
+lambda_map_while_lambda_map_strided_slice_1k
glambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor¢Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertk
(lambda/map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB ²
4lambda/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0lambda_map_while_placeholder1lambda/map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0¬
lambda/map/while/DecodeJpeg
DecodeJpeg;lambda/map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
lambda/map/while/CastCast#lambda/map/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ_
lambda/map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C¢
lambda/map/while/truedivRealDivlambda/map/while/Cast:y:0#lambda/map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
-lambda/map/while/random_flip_left_right/ShapeShapelambda/map/while/truediv:z:0*
T0*
_output_shapes
:
;lambda/map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
=lambda/map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=lambda/map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5lambda/map/while/random_flip_left_right/strided_sliceStridedSlice6lambda/map/while/random_flip_left_right/Shape:output:0Dlambda/map/while/random_flip_left_right/strided_slice/stack:output:0Flambda/map/while/random_flip_left_right/strided_slice/stack_1:output:0Flambda/map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
=lambda/map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ý
Hlambda/map/while/random_flip_left_right/assert_positive/assert_less/LessLessFlambda/map/while/random_flip_left_right/assert_positive/Const:output:0>lambda/map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Ilambda/map/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Glambda/map/while/random_flip_left_right/assert_positive/assert_less/AllAllLlambda/map/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Rlambda/map/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: »
Plambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ã
Xlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertPlambda/map/while/random_flip_left_right/assert_positive/assert_less/All:output:0alambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 n
,lambda/map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :
>lambda/map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :ú
Ilambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual5lambda/map/while/random_flip_left_right/Rank:output:0Glambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
Alambda/map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Hlambda/map/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Hlambda/map/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :×
Blambda/map/while/random_flip_left_right/assert_greater_equal/rangeRangeQlambda/map/while/random_flip_left_right/assert_greater_equal/range/start:output:0Jlambda/map/while/random_flip_left_right/assert_greater_equal/Rank:output:0Qlambda/map/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: û
@lambda/map/while/random_flip_left_right/assert_greater_equal/AllAllMlambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Klambda/map/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: µ
Ilambda/map/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.·
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Á
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*F
value=B; B5x (lambda/map/while/random_flip_left_right/Rank:0) = Ó
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*X
valueOBM BGy (lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = ½
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.½
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Ç
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*F
value=B; B5x (lambda/map/while/random_flip_left_right/Rank:0) = Ù
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*X
valueOBM BGy (lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = 
Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertIlambda/map/while/random_flip_left_right/assert_greater_equal/All:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:05lambda/map/while/random_flip_left_right/Rank:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0Glambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0R^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ñ
:lambda/map/while/random_flip_left_right/control_dependencyIdentitylambda/map/while/truediv:z:0K^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertR^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*+
_class!
loc:@lambda/map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<lambda/map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
:lambda/map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:lambda/map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ê
Dlambda/map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniformElambda/map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0ö
:lambda/map/while/random_flip_left_right/random_uniform/MulMulMlambda/map/while/random_flip_left_right/random_uniform/RandomUniform:output:0Clambda/map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: s
.lambda/map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Î
,lambda/map/while/random_flip_left_right/LessLess>lambda/map/while/random_flip_left_right/random_uniform/Mul:z:07lambda/map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 
'lambda/map/while/random_flip_left_rightStatelessIf0lambda/map/while/random_flip_left_right/Less:z:0Clambda/map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *G
else_branch8R6
4lambda_map_while_random_flip_left_right_false_157562*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*F
then_branch7R5
3lambda_map_while_random_flip_left_right_true_157561­
0lambda/map/while/random_flip_left_right/IdentityIdentity0lambda/map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*lambda/map/while/random_flip_up_down/ShapeShape9lambda/map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
8lambda/map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:lambda/map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:lambda/map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2lambda/map/while/random_flip_up_down/strided_sliceStridedSlice3lambda/map/while/random_flip_up_down/Shape:output:0Alambda/map/while/random_flip_up_down/strided_slice/stack:output:0Clambda/map/while/random_flip_up_down/strided_slice/stack_1:output:0Clambda/map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask|
:lambda/map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ô
Elambda/map/while/random_flip_up_down/assert_positive/assert_less/LessLessClambda/map/while/random_flip_up_down/assert_positive/Const:output:0;lambda/map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
Flambda/map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ÿ
Dlambda/map/while/random_flip_up_down/assert_positive/assert_less/AllAllIlambda/map/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Olambda/map/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ¸
Mlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.À
Ulambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.ö
Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertMlambda/map/while/random_flip_up_down/assert_positive/assert_less/All:output:0^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0K^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 k
)lambda/map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;lambda/map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :ñ
Flambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual2lambda/map/while/random_flip_up_down/Rank:output:0Dlambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
>lambda/map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Elambda/map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Elambda/map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ë
?lambda/map/while/random_flip_up_down/assert_greater_equal/rangeRangeNlambda/map/while/random_flip_up_down/assert_greater_equal/range/start:output:0Glambda/map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Nlambda/map/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: ò
=lambda/map/while/random_flip_up_down/assert_greater_equal/AllAllJlambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Hlambda/map/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: ²
Flambda/map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.´
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:»
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2x (lambda/map/while/random_flip_up_down/Rank:0) = Í
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = º
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.º
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Á
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2x (lambda/map/while/random_flip_up_down/Rank:0) = Ó
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = ï
Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssertFlambda/map/while/random_flip_up_down/assert_greater_equal/All:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:02lambda/map/while/random_flip_up_down/Rank:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0Dlambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0O^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 
7lambda/map/while/random_flip_up_down/control_dependencyIdentity9lambda/map/while/random_flip_left_right/Identity:output:0H^lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertO^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*C
_class9
75loc:@lambda/map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
9lambda/map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB |
7lambda/map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    |
7lambda/map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
Alambda/map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniformBlambda/map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0í
7lambda/map/while/random_flip_up_down/random_uniform/MulMulJlambda/map/while/random_flip_up_down/random_uniform/RandomUniform:output:0@lambda/map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: p
+lambda/map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
)lambda/map/while/random_flip_up_down/LessLess;lambda/map/while/random_flip_up_down/random_uniform/Mul:z:04lambda/map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 
$lambda/map/while/random_flip_up_downStatelessIf-lambda/map/while/random_flip_up_down/Less:z:0@lambda/map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *D
else_branch5R3
1lambda_map_while_random_flip_up_down_false_157609*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*C
then_branch4R2
0lambda_map_while_random_flip_up_down_true_157608§
-lambda/map/while/random_flip_up_down/IdentityIdentity-lambda/map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
/lambda/map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB r
-lambda/map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½r
-lambda/map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Klambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      ë
Flambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterTlambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Flambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :¡
Blambda/map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV28lambda/map/while/stateless_random_uniform/shape:output:0Llambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Plambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Olambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: Å
-lambda/map/while/stateless_random_uniform/subSub6lambda/map/while/stateless_random_uniform/max:output:06lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Õ
-lambda/map/while/stateless_random_uniform/mulMulKlambda/map/while/stateless_random_uniform/StatelessRandomUniformV2:output:01lambda/map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ¾
)lambda/map/while/stateless_random_uniformAddV21lambda/map/while/stateless_random_uniform/mul:z:06lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ñ
"lambda/map/while/adjust_brightnessAddV26lambda/map/while/random_flip_up_down/Identity:output:0-lambda/map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+lambda/map/while/adjust_brightness/IdentityIdentity&lambda/map/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
%lambda/map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB h
#lambda/map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
#lambda/map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
-lambda/map/while/random_uniform/RandomUniformRandomUniform.lambda/map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0§
#lambda/map/while/random_uniform/subSub,lambda/map/while/random_uniform/max:output:0,lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: ¬
#lambda/map/while/random_uniform/mulMul6lambda/map/while/random_uniform/RandomUniform:output:0'lambda/map/while/random_uniform/sub:z:0*
T0*
_output_shapes
:  
lambda/map/while/random_uniformAddV2'lambda/map/while/random_uniform/mul:z:0,lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ø
3lambda/map/while/adjust_saturation/AdjustSaturationAdjustSaturation4lambda/map/while/adjust_brightness/Identity:output:0#lambda/map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
+lambda/map/while/adjust_saturation/IdentityIdentity<lambda/map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
&lambda/map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ú
"lambda/map/while/resize/ExpandDims
ExpandDims4lambda/map/while/adjust_saturation/Identity:output:0/lambda/map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿm
lambda/map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ù
&lambda/map/while/resize/ResizeBilinearResizeBilinear+lambda/map/while/resize/ExpandDims:output:0%lambda/map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(©
lambda/map/while/resize/SqueezeSqueeze7lambda/map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 ò
5lambda/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlambda_map_while_placeholder_1lambda_map_while_placeholder(lambda/map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒX
lambda/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :}
lambda/map/while/addAddV2lambda_map_while_placeholderlambda/map/while/add/y:output:0*
T0*
_output_shapes
: Z
lambda/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lambda/map/while/add_1AddV2.lambda_map_while_lambda_map_while_loop_counter!lambda/map/while/add_1/y:output:0*
T0*
_output_shapes
: z
lambda/map/while/IdentityIdentitylambda/map/while/add_1:z:0^lambda/map/while/NoOp*
T0*
_output_shapes
: 
lambda/map/while/Identity_1Identity)lambda_map_while_lambda_map_strided_slice^lambda/map/while/NoOp*
T0*
_output_shapes
: z
lambda/map/while/Identity_2Identitylambda/map/while/add:z:0^lambda/map/while/NoOp*
T0*
_output_shapes
: º
lambda/map/while/Identity_3IdentityElambda/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lambda/map/while/NoOp*
T0*
_output_shapes
: :éèÒ
lambda/map/while/NoOpNoOpK^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertR^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertH^lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertO^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
lambda_map_while_identity"lambda/map/while/Identity:output:0"C
lambda_map_while_identity_1$lambda/map/while/Identity_1:output:0"C
lambda_map_while_identity_2$lambda/map/while/Identity_2:output:0"C
lambda_map_while_identity_3$lambda/map/while/Identity_3:output:0"\
+lambda_map_while_lambda_map_strided_slice_1-lambda_map_while_lambda_map_strided_slice_1_0"Ô
glambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensorilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2
Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertJlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert2¦
Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertQlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertGlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2 
Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertNlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Õ
Ý
4lambda_map_while_random_flip_left_right_false_157208o
klambda_map_while_random_flip_left_right_identity_lambda_map_while_random_flip_left_right_control_dependency4
0lambda_map_while_random_flip_left_right_identityè
0lambda/map/while/random_flip_left_right/IdentityIdentityklambda_map_while_random_flip_left_right_identity_lambda_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"m
0lambda_map_while_random_flip_left_right_identity9lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­
¦
map_while_body_157865$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor¢9map/while/central_crop/assert_greater_equal/Assert/Assert¢@map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Cmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertd
!map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB 
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholder*map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0
map/while/DecodeJpeg
DecodeJpeg4map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
map/while/CastCastmap/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿX
map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
map/while/truedivRealDivmap/while/Cast:y:0map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
&map/while/random_flip_left_right/ShapeShapemap/while/truediv:z:0*
T0*
_output_shapes
:
4map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
.map/while/random_flip_left_right/strided_sliceStridedSlice/map/while/random_flip_left_right/Shape:output:0=map/while/random_flip_left_right/strided_slice/stack:output:0?map/while/random_flip_left_right/strided_slice/stack_1:output:0?map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskx
6map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : è
Amap/while/random_flip_left_right/assert_positive/assert_less/LessLess?map/while/random_flip_left_right/assert_positive/Const:output:07map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Bmap/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ó
@map/while/random_flip_left_right/assert_positive/assert_less/AllAllEmap/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Kmap/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: ´
Imap/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¼
Qmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertImap/while/random_flip_left_right/assert_positive/assert_less/All:output:0Zmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 g
%map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :y
7map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :å
Bmap/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.map/while/random_flip_left_right/Rank:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: |
:map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :»
;map/while/random_flip_left_right/assert_greater_equal/rangeRangeJmap/while/random_flip_left_right/assert_greater_equal/range/start:output:0Cmap/while/random_flip_left_right/assert_greater_equal/Rank:output:0Jmap/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: æ
9map/while/random_flip_left_right/assert_greater_equal/AllAllFmap/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dmap/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: ®
Bmap/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.°
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Å
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = ¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¹
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Ë
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = Ë
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertBmap/while/random_flip_left_right/assert_greater_equal/All:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.map/while/random_flip_left_right/Rank:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0K^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Î
3map/while/random_flip_left_right/control_dependencyIdentitymap/while/truediv:z:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*$
_class
loc:@map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
5map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB x
3map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
=map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniform>map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0á
3map/while/random_flip_left_right/random_uniform/MulMulFmap/while/random_flip_left_right/random_uniform/RandomUniform:output:0<map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: l
'map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
%map/while/random_flip_left_right/LessLess7map/while/random_flip_left_right/random_uniform/Mul:z:00map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ó
 map/while/random_flip_left_rightStatelessIf)map/while/random_flip_left_right/Less:z:0<map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-map_while_random_flip_left_right_false_157926*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,map_while_random_flip_left_right_true_157925
)map/while/random_flip_left_right/IdentityIdentity)map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
#map/while/random_flip_up_down/ShapeShape2map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
1map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
3map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
+map/while/random_flip_up_down/strided_sliceStridedSlice,map/while/random_flip_up_down/Shape:output:0:map/while/random_flip_up_down/strided_slice/stack:output:0<map/while/random_flip_up_down/strided_slice/stack_1:output:0<map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masku
3map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ß
>map/while/random_flip_up_down/assert_positive/assert_less/LessLess<map/while/random_flip_up_down/assert_positive/Const:output:04map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
?map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ê
=map/while/random_flip_up_down/assert_positive/assert_less/AllAllBmap/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Hmap/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ±
Fmap/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¹
Nmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ú
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFmap/while/random_flip_up_down/assert_positive/assert_less/All:output:0Wmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 d
"map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :v
4map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ü
?map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+map/while/random_flip_up_down/Rank:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: y
7map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¯
8map/while/random_flip_up_down/assert_greater_equal/rangeRangeGmap/while/random_flip_up_down/assert_greater_equal/range/start:output:0@map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Gmap/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: Ý
6map/while/random_flip_up_down/assert_greater_equal/AllAllCmap/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Amap/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: «
?map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = ¿
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = ³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = Å
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = °
@map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssert?map/while/random_flip_up_down/assert_greater_equal/All:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+map/while/random_flip_up_down/Rank:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0H^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ú
0map/while/random_flip_up_down/control_dependencyIdentity2map/while/random_flip_left_right/Identity:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
2map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB u
0map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
:map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniform;map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0Ø
0map/while/random_flip_up_down/random_uniform/MulMulCmap/while/random_flip_up_down/random_uniform/RandomUniform:output:09map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: i
$map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
"map/while/random_flip_up_down/LessLess4map/while/random_flip_up_down/random_uniform/Mul:z:0-map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: ä
map/while/random_flip_up_downStatelessIf&map/while/random_flip_up_down/Less:z:09map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*map_while_random_flip_up_down_false_157973*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)map_while_random_flip_up_down_true_157972
&map/while/random_flip_up_down/IdentityIdentity&map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
(map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½k
&map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Dmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      Ý
?map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterMmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
?map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21map/while/stateless_random_uniform/shape:output:0Emap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Imap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hmap/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&map/while/stateless_random_uniform/subSub/map/while/stateless_random_uniform/max:output:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&map/while/stateless_random_uniform/mulMulDmap/while/stateless_random_uniform/StatelessRandomUniformV2:output:0*map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"map/while/stateless_random_uniformAddV2*map/while/stateless_random_uniform/mul:z:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
map/while/adjust_brightnessAddV2/map/while/random_flip_up_down/Identity:output:0&map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$map/while/adjust_brightness/IdentityIdentitymap/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
&map/while/random_uniform/RandomUniformRandomUniform'map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0
map/while/random_uniform/subSub%map/while/random_uniform/max:output:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: 
map/while/random_uniform/mulMul/map/while/random_uniform/RandomUniform:output:0 map/while/random_uniform/sub:z:0*
T0*
_output_shapes
: 
map/while/random_uniformAddV2 map/while/random_uniform/mul:z:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ã
,map/while/adjust_saturation/AdjustSaturationAdjustSaturation-map/while/adjust_brightness/Identity:output:0map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
$map/while/adjust_saturation/IdentityIdentity5map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
map/while/central_crop/ShapeShape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
*map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
,map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$map/while/central_crop/strided_sliceStridedSlice%map/while/central_crop/Shape:output:03map/while/central_crop/strided_slice/stack:output:05map/while/central_crop/strided_slice/stack_1:output:05map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
,map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ê
7map/while/central_crop/assert_positive/assert_less/LessLess5map/while/central_crop/assert_positive/Const:output:0-map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
8map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: Õ
6map/while/central_crop/assert_positive/assert_less/AllAll;map/while/central_crop/assert_positive/assert_less/Less:z:0Amap/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ª
?map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Gmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Â
@map/while/central_crop/assert_positive/assert_less/Assert/AssertAssert?map/while/central_crop/assert_positive/assert_less/All:output:0Pmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 ]
map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ç
8map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual$map/while/central_crop/Rank:output:06map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: r
0map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
1map/while/central_crop/assert_greater_equal/rangeRange@map/while/central_crop/assert_greater_equal/range/start:output:09map/while/central_crop/assert_greater_equal/Rank:output:0@map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: È
/map/while/central_crop/assert_greater_equal/AllAll<map/while/central_crop/assert_greater_equal/GreaterEqual:z:0:map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: ¤
8map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¦
:map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
:map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ±
:map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¥
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ·
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ñ
9map/while/central_crop/assert_greater_equal/Assert/AssertAssert8map/while/central_crop/assert_greater_equal/All:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:0$map/while/central_crop/Rank:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:06map/while/central_crop/assert_greater_equal/y:output:0A^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Û
)map/while/central_crop/control_dependencyIdentity-map/while/adjust_saturation/Identity:output:0:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*7
_class-
+)loc:@map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
map/while/central_crop/Shape_1Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_1StridedSlice'map/while/central_crop/Shape_1:output:05map/while/central_crop/strided_slice_1/stack:output:07map/while/central_crop/strided_slice_1/stack_1:output:07map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
map/while/central_crop/Shape_2Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_2StridedSlice'map/while/central_crop/Shape_2:output:05map/while/central_crop/strided_slice_2/stack:output:07map/while/central_crop/strided_slice_2/stack_1:output:07map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
map/while/central_crop/CastCast/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *Â)?
map/while/central_crop/Cast_1Cast(map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mulMulmap/while/central_crop/Cast:y:0!map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: 
map/while/central_crop/subSubmap/while/central_crop/Cast:y:0map/while/central_crop/mul:z:0*
T0*
_output_shapes
: i
 map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
map/while/central_crop/truedivRealDivmap/while/central_crop/sub:z:0)map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: y
map/while/central_crop/Cast_2Cast"map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/Cast_3Cast/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *Â)?
map/while/central_crop/Cast_4Cast(map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mul_1Mul!map/while/central_crop/Cast_3:y:0!map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_1Sub!map/while/central_crop/Cast_3:y:0 map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: k
"map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
 map/while/central_crop/truediv_1RealDiv map/while/central_crop/sub_1:z:0+map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: {
map/while/central_crop/Cast_5Cast$map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: `
map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_2Mul!map/while/central_crop/Cast_2:y:0'map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_2Sub/map/while/central_crop/strided_slice_1:output:0 map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: `
map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_3Mul!map/while/central_crop/Cast_5:y:0'map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_3Sub/map/while/central_crop/strided_slice_2:output:0 map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: `
map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Á
map/while/central_crop/stackPack!map/while/central_crop/Cast_2:y:0!map/while/central_crop/Cast_5:y:0'map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:k
 map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
map/while/central_crop/stack_1Pack map/while/central_crop/sub_2:z:0 map/while/central_crop/sub_3:z:0)map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:ð
map/while/central_crop/SliceSlice-map/while/adjust_saturation/Identity:output:0%map/while/central_crop/stack:output:0'map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
map/while/resize/ExpandDims
ExpandDims%map/while/central_crop/Slice:output:0(map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ä
map/while/resize/ResizeBilinearResizeBilinear$map/while/resize/ExpandDims:output:0map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(
map/while/resize/SqueezeSqueeze0map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 Ö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder!map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒï
map/while/NoOpNoOp:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/AssertD^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertA^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2v
9map/while/central_crop/assert_greater_equal/Assert/Assert9map/while/central_crop/assert_greater_equal/Assert/Assert2
@map/while/central_crop/assert_positive/assert_less/Assert/Assert@map/while/central_crop/assert_positive/assert_less/Assert/Assert2
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertCmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert2
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertJmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertGmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ý
¢
-__inference_block1_conv1_layer_call_fn_159072

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ü
¥
-__inference_block4_conv1_layer_call_fn_159242

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
_
C__inference_CLASSES_layer_call_and_return_conditional_losses_158725

inputs
identity	R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :a
ArgMaxArgMaxinputsArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentityArgMax:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

map_while_cond_155965$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_155965___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ø

map_while_cond_157864$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_157864___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Ö
Ý
3lambda_map_while_random_flip_left_right_true_157561p
llambda_map_while_random_flip_left_right_reversev2_lambda_map_while_random_flip_left_right_control_dependency4
0lambda_map_while_random_flip_left_right_identity
6lambda/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:¬
1lambda/map/while/random_flip_left_right/ReverseV2	ReverseV2llambda_map_while_random_flip_left_right_reversev2_lambda_map_while_random_flip_left_right_control_dependency?lambda/map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
0lambda/map/while/random_flip_left_right/IdentityIdentity:lambda/map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"m
0lambda_map_while_random_flip_left_right_identity9lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¨Z
¦
A__inference_vgg16_layer_call_and_return_conditional_losses_155427
input_1-
block1_conv1_155355:@!
block1_conv1_155357:@-
block1_conv2_155360:@@!
block1_conv2_155362:@.
block2_conv1_155366:@"
block2_conv1_155368:	/
block2_conv2_155371:"
block2_conv2_155373:	/
block3_conv1_155377:"
block3_conv1_155379:	/
block3_conv2_155382:"
block3_conv2_155384:	/
block3_conv3_155387:"
block3_conv3_155389:	/
block4_conv1_155393:"
block4_conv1_155395:	/
block4_conv2_155398:"
block4_conv2_155400:	/
block4_conv3_155403:"
block4_conv3_155405:	/
block5_conv1_155409:"
block5_conv1_155411:	/
block5_conv2_155414:"
block5_conv2_155416:	/
block5_conv3_155419:"
block5_conv3_155421:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_155355block1_conv1_155357*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_155360block1_conv2_155362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_155366block2_conv1_155368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_155371block2_conv2_155373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_155377block3_conv1_155379*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_155382block3_conv2_155384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_155387block3_conv3_155389*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_155393block4_conv1_155395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_155398block4_conv2_155400*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_155403block4_conv3_155405*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_155409block5_conv1_155411*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_155414block5_conv2_155416*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_155419block5_conv3_155421*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595ú
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
ª
J
.__inference_PROBABILITIES_layer_call_fn_158690

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156259`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
e
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158699

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
¦
map_while_body_155966$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor¢9map/while/central_crop/assert_greater_equal/Assert/Assert¢@map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Cmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertd
!map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB 
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholder*map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0
map/while/DecodeJpeg
DecodeJpeg4map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
map/while/CastCastmap/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿX
map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
map/while/truedivRealDivmap/while/Cast:y:0map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
&map/while/random_flip_left_right/ShapeShapemap/while/truediv:z:0*
T0*
_output_shapes
:
4map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
6map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
.map/while/random_flip_left_right/strided_sliceStridedSlice/map/while/random_flip_left_right/Shape:output:0=map/while/random_flip_left_right/strided_slice/stack:output:0?map/while/random_flip_left_right/strided_slice/stack_1:output:0?map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskx
6map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : è
Amap/while/random_flip_left_right/assert_positive/assert_less/LessLess?map/while/random_flip_left_right/assert_positive/Const:output:07map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Bmap/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ó
@map/while/random_flip_left_right/assert_positive/assert_less/AllAllEmap/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Kmap/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: ´
Imap/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¼
Qmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertImap/while/random_flip_left_right/assert_positive/assert_less/All:output:0Zmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 g
%map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :y
7map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :å
Bmap/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.map/while/random_flip_left_right/Rank:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: |
:map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Amap/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :»
;map/while/random_flip_left_right/assert_greater_equal/rangeRangeJmap/while/random_flip_left_right/assert_greater_equal/range/start:output:0Cmap/while/random_flip_left_right/assert_greater_equal/Rank:output:0Jmap/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: æ
9map/while/random_flip_left_right/assert_greater_equal/AllAllFmap/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dmap/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: ®
Bmap/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.°
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Å
Dmap/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = ¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¶
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¹
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (map/while/random_flip_left_right/Rank:0) = Ë
Jmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (map/while/random_flip_left_right/assert_greater_equal/y:0) = Ë
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertBmap/while/random_flip_left_right/assert_greater_equal/All:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.map/while/random_flip_left_right/Rank:output:0Smap/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@map/while/random_flip_left_right/assert_greater_equal/y:output:0K^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Î
3map/while/random_flip_left_right/control_dependencyIdentitymap/while/truediv:z:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*$
_class
loc:@map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
5map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB x
3map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
=map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniform>map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0á
3map/while/random_flip_left_right/random_uniform/MulMulFmap/while/random_flip_left_right/random_uniform/RandomUniform:output:0<map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: l
'map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¹
%map/while/random_flip_left_right/LessLess7map/while/random_flip_left_right/random_uniform/Mul:z:00map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ó
 map/while/random_flip_left_rightStatelessIf)map/while/random_flip_left_right/Less:z:0<map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-map_while_random_flip_left_right_false_156027*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,map_while_random_flip_left_right_true_156026
)map/while/random_flip_left_right/IdentityIdentity)map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
#map/while/random_flip_up_down/ShapeShape2map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
1map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
3map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
+map/while/random_flip_up_down/strided_sliceStridedSlice,map/while/random_flip_up_down/Shape:output:0:map/while/random_flip_up_down/strided_slice/stack:output:0<map/while/random_flip_up_down/strided_slice/stack_1:output:0<map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masku
3map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ß
>map/while/random_flip_up_down/assert_positive/assert_less/LessLess<map/while/random_flip_up_down/assert_positive/Const:output:04map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
?map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ê
=map/while/random_flip_up_down/assert_positive/assert_less/AllAllBmap/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Hmap/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ±
Fmap/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¹
Nmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ú
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFmap/while/random_flip_up_down/assert_positive/assert_less/All:output:0Wmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0D^map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 d
"map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :v
4map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ü
?map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+map/while/random_flip_up_down/Rank:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: y
7map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
>map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¯
8map/while/random_flip_up_down/assert_greater_equal/rangeRangeGmap/while/random_flip_up_down/assert_greater_equal/range/start:output:0@map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Gmap/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: Ý
6map/while/random_flip_up_down/assert_greater_equal/AllAllCmap/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Amap/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: «
?map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:­
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = ¿
Amap/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = ³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (map/while/random_flip_up_down/Rank:0) = Å
Gmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (map/while/random_flip_up_down/assert_greater_equal/y:0) = °
@map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssert?map/while/random_flip_up_down/assert_greater_equal/All:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+map/while/random_flip_up_down/Rank:output:0Pmap/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=map/while/random_flip_up_down/assert_greater_equal/y:output:0H^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ú
0map/while/random_flip_up_down/control_dependencyIdentity2map/while/random_flip_left_right/Identity:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿu
2map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB u
0map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
:map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniform;map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0Ø
0map/while/random_flip_up_down/random_uniform/MulMulCmap/while/random_flip_up_down/random_uniform/RandomUniform:output:09map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: i
$map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?°
"map/while/random_flip_up_down/LessLess4map/while/random_flip_up_down/random_uniform/Mul:z:0-map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: ä
map/while/random_flip_up_downStatelessIf&map/while/random_flip_up_down/Less:z:09map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*map_while_random_flip_up_down_false_156074*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)map_while_random_flip_up_down_true_156073
&map/while/random_flip_up_down/IdentityIdentity&map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿk
(map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½k
&map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Dmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      Ý
?map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterMmap/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
?map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21map/while/stateless_random_uniform/shape:output:0Emap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Imap/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hmap/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&map/while/stateless_random_uniform/subSub/map/while/stateless_random_uniform/max:output:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&map/while/stateless_random_uniform/mulMulDmap/while/stateless_random_uniform/StatelessRandomUniformV2:output:0*map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"map/while/stateless_random_uniformAddV2*map/while/stateless_random_uniform/mul:z:0/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
map/while/adjust_brightnessAddV2/map/while/random_flip_up_down/Identity:output:0&map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$map/while/adjust_brightness/IdentityIdentitymap/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
&map/while/random_uniform/RandomUniformRandomUniform'map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0
map/while/random_uniform/subSub%map/while/random_uniform/max:output:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: 
map/while/random_uniform/mulMul/map/while/random_uniform/RandomUniform:output:0 map/while/random_uniform/sub:z:0*
T0*
_output_shapes
: 
map/while/random_uniformAddV2 map/while/random_uniform/mul:z:0%map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ã
,map/while/adjust_saturation/AdjustSaturationAdjustSaturation-map/while/adjust_brightness/Identity:output:0map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
$map/while/adjust_saturation/IdentityIdentity5map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
map/while/central_crop/ShapeShape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
*map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
,map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$map/while/central_crop/strided_sliceStridedSlice%map/while/central_crop/Shape:output:03map/while/central_crop/strided_slice/stack:output:05map/while/central_crop/strided_slice/stack_1:output:05map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskn
,map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ê
7map/while/central_crop/assert_positive/assert_less/LessLess5map/while/central_crop/assert_positive/Const:output:0-map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
8map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: Õ
6map/while/central_crop/assert_positive/assert_less/AllAll;map/while/central_crop/assert_positive/assert_less/Less:z:0Amap/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ª
?map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Gmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Â
@map/while/central_crop/assert_positive/assert_less/Assert/AssertAssert?map/while/central_crop/assert_positive/assert_less/All:output:0Pmap/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0A^map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 ]
map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ç
8map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual$map/while/central_crop/Rank:output:06map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: r
0map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : y
7map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
1map/while/central_crop/assert_greater_equal/rangeRange@map/while/central_crop/assert_greater_equal/range/start:output:09map/while/central_crop/assert_greater_equal/Rank:output:0@map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: È
/map/while/central_crop/assert_greater_equal/AllAll<map/while/central_crop/assert_greater_equal/GreaterEqual:z:0:map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: ¤
8map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¦
:map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:
:map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ±
:map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¬
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¥
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (map/while/central_crop/Rank:0) = ·
@map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (map/while/central_crop/assert_greater_equal/y:0) = ñ
9map/while/central_crop/assert_greater_equal/Assert/AssertAssert8map/while/central_crop/assert_greater_equal/All:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:0$map/while/central_crop/Rank:output:0Imap/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:06map/while/central_crop/assert_greater_equal/y:output:0A^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 Û
)map/while/central_crop/control_dependencyIdentity-map/while/adjust_saturation/Identity:output:0:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*7
_class-
+)loc:@map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
map/while/central_crop/Shape_1Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_1StridedSlice'map/while/central_crop/Shape_1:output:05map/while/central_crop/strided_slice_1/stack:output:07map/while/central_crop/strided_slice_1/stack_1:output:07map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
map/while/central_crop/Shape_2Shape-map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:v
,map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
&map/while/central_crop/strided_slice_2StridedSlice'map/while/central_crop/Shape_2:output:05map/while/central_crop/strided_slice_2/stack:output:07map/while/central_crop/strided_slice_2/stack_1:output:07map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
map/while/central_crop/CastCast/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *&c?
map/while/central_crop/Cast_1Cast(map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mulMulmap/while/central_crop/Cast:y:0!map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: 
map/while/central_crop/subSubmap/while/central_crop/Cast:y:0map/while/central_crop/mul:z:0*
T0*
_output_shapes
: i
 map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
map/while/central_crop/truedivRealDivmap/while/central_crop/sub:z:0)map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: y
map/while/central_crop/Cast_2Cast"map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/Cast_3Cast/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *&c?
map/while/central_crop/Cast_4Cast(map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
map/while/central_crop/mul_1Mul!map/while/central_crop/Cast_3:y:0!map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_1Sub!map/while/central_crop/Cast_3:y:0 map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: k
"map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @
 map/while/central_crop/truediv_1RealDiv map/while/central_crop/sub_1:z:0+map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: {
map/while/central_crop/Cast_5Cast$map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: `
map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_2Mul!map/while/central_crop/Cast_2:y:0'map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_2Sub/map/while/central_crop/strided_slice_1:output:0 map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: `
map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :
map/while/central_crop/mul_3Mul!map/while/central_crop/Cast_5:y:0'map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: 
map/while/central_crop/sub_3Sub/map/while/central_crop/strided_slice_2:output:0 map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: `
map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Á
map/while/central_crop/stackPack!map/while/central_crop/Cast_2:y:0!map/while/central_crop/Cast_5:y:0'map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:k
 map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÃ
map/while/central_crop/stack_1Pack map/while/central_crop/sub_2:z:0 map/while/central_crop/sub_3:z:0)map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:ð
map/while/central_crop/SliceSlice-map/while/adjust_saturation/Identity:output:0%map/while/central_crop/stack:output:0'map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
map/while/resize/ExpandDims
ExpandDims%map/while/central_crop/Slice:output:0(map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿf
map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ä
map/while/resize/ResizeBilinearResizeBilinear$map/while/resize/ExpandDims:output:0map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(
map/while/resize/SqueezeSqueeze0map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 Ö
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder!map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: e
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: o
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: e
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: ¥
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: :éèÒï
map/while/NoOpNoOp:^map/while/central_crop/assert_greater_equal/Assert/AssertA^map/while/central_crop/assert_positive/assert_less/Assert/AssertD^map/while/random_flip_left_right/assert_greater_equal/Assert/AssertK^map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertA^map/while/random_flip_up_down/assert_greater_equal/Assert/AssertH^map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2v
9map/while/central_crop/assert_greater_equal/Assert/Assert9map/while/central_crop/assert_greater_equal/Assert/Assert2
@map/while/central_crop/assert_positive/assert_less/Assert/Assert@map/while/central_crop/assert_positive/assert_less/Assert/Assert2
Cmap/while/random_flip_left_right/assert_greater_equal/Assert/AssertCmap/while/random_flip_left_right/assert_greater_equal/Assert/Assert2
Jmap/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertJmap/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert@map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2
Gmap/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertGmap/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

c
G__inference_block1_pool_layer_call_and_return_conditional_losses_159113

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv2_layer_call_and_return_conditional_losses_159153

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
®
Ñ
0lambda_map_while_random_flip_up_down_true_157254j
flambda_map_while_random_flip_up_down_reversev2_lambda_map_while_random_flip_up_down_control_dependency1
-lambda_map_while_random_flip_up_down_identity}
3lambda/map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:  
.lambda/map/while/random_flip_up_down/ReverseV2	ReverseV2flambda_map_while_random_flip_up_down_reversev2_lambda_map_while_random_flip_up_down_control_dependency<lambda/map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
-lambda/map/while/random_flip_up_down/IdentityIdentity7lambda/map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"g
-lambda_map_while_random_flip_up_down_identity6lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
º
Ñ
1lambda_map_while_random_flip_up_down_false_157255i
elambda_map_while_random_flip_up_down_identity_lambda_map_while_random_flip_up_down_control_dependency1
-lambda_map_while_random_flip_up_down_identityß
-lambda/map/while/random_flip_up_down/IdentityIdentityelambda_map_while_random_flip_up_down_identity_lambda_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"g
-lambda_map_while_random_flip_up_down_identity6lambda/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ü
¥
-__inference_block2_conv2_layer_call_fn_159142

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿpp: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs

U
9__inference_global_average_pooling2d_layer_call_fn_159378

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv3_layer_call_fn_159352

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥Ê
ë	
"model_lambda_map_while_body_154200>
:model_lambda_map_while_model_lambda_map_while_loop_counter9
5model_lambda_map_while_model_lambda_map_strided_slice&
"model_lambda_map_while_placeholder(
$model_lambda_map_while_placeholder_1=
9model_lambda_map_while_model_lambda_map_strided_slice_1_0y
umodel_lambda_map_while_tensorarrayv2read_tensorlistgetitem_model_lambda_map_tensorarrayunstack_tensorlistfromtensor_0#
model_lambda_map_while_identity%
!model_lambda_map_while_identity_1%
!model_lambda_map_while_identity_2%
!model_lambda_map_while_identity_3;
7model_lambda_map_while_model_lambda_map_strided_slice_1w
smodel_lambda_map_while_tensorarrayv2read_tensorlistgetitem_model_lambda_map_tensorarrayunstack_tensorlistfromtensor¢Fmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert¢Mmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Pmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Wmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Mmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Tmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertq
.model/lambda/map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB Ð
:model/lambda/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemumodel_lambda_map_while_tensorarrayv2read_tensorlistgetitem_model_lambda_map_tensorarrayunstack_tensorlistfromtensor_0"model_lambda_map_while_placeholder7model/lambda/map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0¸
!model/lambda/map/while/DecodeJpeg
DecodeJpegAmodel/lambda/map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
model/lambda/map/while/CastCast)model/lambda/map/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿe
 model/lambda/map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C´
model/lambda/map/while/truedivRealDivmodel/lambda/map/while/Cast:y:0)model/lambda/map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
3model/lambda/map/while/random_flip_left_right/ShapeShape"model/lambda/map/while/truediv:z:0*
T0*
_output_shapes
:
Amodel/lambda/map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
Cmodel/lambda/map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Cmodel/lambda/map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
;model/lambda/map/while/random_flip_left_right/strided_sliceStridedSlice<model/lambda/map/while/random_flip_left_right/Shape:output:0Jmodel/lambda/map/while/random_flip_left_right/strided_slice/stack:output:0Lmodel/lambda/map/while/random_flip_left_right/strided_slice/stack_1:output:0Lmodel/lambda/map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Cmodel/lambda/map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Nmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/LessLessLmodel/lambda/map/while/random_flip_left_right/assert_positive/Const:output:0Dmodel/lambda/map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Omodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Mmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/AllAllRmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Xmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: Á
Vmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.É
^model/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ä
Wmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertVmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/All:output:0gmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 t
2model/lambda/map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :
Dmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :
Omodel/lambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual;model/lambda/map/while/random_flip_left_right/Rank:output:0Mmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
Gmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Nmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Nmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ï
Hmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/rangeRangeWmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/range/start:output:0Pmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Rank:output:0Wmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 
Fmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/AllAllSmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Qmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: »
Omodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.½
Qmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Í
Qmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*L
valueCBA B;x (model/lambda/map/while/random_flip_left_right/Rank:0) = ß
Qmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*^
valueUBS BMy (model/lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = Ã
Wmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.Ã
Wmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Ó
Wmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*L
valueCBA B;x (model/lambda/map/while/random_flip_left_right/Rank:0) = å
Wmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*^
valueUBS BMy (model/lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = À
Pmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertOmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/All:output:0`model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0`model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0`model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0;model/lambda/map/while/random_flip_left_right/Rank:output:0`model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0Mmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0X^model/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 
@model/lambda/map/while/random_flip_left_right/control_dependencyIdentity"model/lambda/map/while/truediv:z:0Q^model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertX^model/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*1
_class'
%#loc:@model/lambda/map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Bmodel/lambda/map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
@model/lambda/map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
@model/lambda/map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ö
Jmodel/lambda/map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniformKmodel/lambda/map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0
@model/lambda/map/while/random_flip_left_right/random_uniform/MulMulSmodel/lambda/map/while/random_flip_left_right/random_uniform/RandomUniform:output:0Imodel/lambda/map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: y
4model/lambda/map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?à
2model/lambda/map/while/random_flip_left_right/LessLessDmodel/lambda/map/while/random_flip_left_right/random_uniform/Mul:z:0=model/lambda/map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: ´
-model/lambda/map/while/random_flip_left_rightStatelessIf6model/lambda/map/while/random_flip_left_right/Less:z:0Imodel/lambda/map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *M
else_branch>R<
:model_lambda_map_while_random_flip_left_right_false_154261*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*L
then_branch=R;
9model_lambda_map_while_random_flip_left_right_true_154260¹
6model/lambda/map/while/random_flip_left_right/IdentityIdentity6model/lambda/map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
0model/lambda/map/while/random_flip_up_down/ShapeShape?model/lambda/map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
>model/lambda/map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
@model/lambda/map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
@model/lambda/map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
8model/lambda/map/while/random_flip_up_down/strided_sliceStridedSlice9model/lambda/map/while/random_flip_up_down/Shape:output:0Gmodel/lambda/map/while/random_flip_up_down/strided_slice/stack:output:0Imodel/lambda/map/while/random_flip_up_down/strided_slice/stack_1:output:0Imodel/lambda/map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@model/lambda/map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Kmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/LessLessImodel/lambda/map/while/random_flip_up_down/assert_positive/Const:output:0Amodel/lambda/map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
Lmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Jmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/AllAllOmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Umodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ¾
Smodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Æ
[model/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.
Tmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertSmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/All:output:0dmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0Q^model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 q
/model/lambda/map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :
Amodel/lambda/map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :
Lmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual8model/lambda/map/while/random_flip_up_down/Rank:output:0Jmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
Dmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Kmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Kmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ã
Emodel/lambda/map/while/random_flip_up_down/assert_greater_equal/rangeRangeTmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/range/start:output:0Mmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Tmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: 
Cmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/AllAllPmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Nmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: ¸
Lmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.º
Nmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Ç
Nmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*I
value@B> B8x (model/lambda/map/while/random_flip_up_down/Rank:0) = Ù
Nmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*[
valueRBP BJy (model/lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = À
Tmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.À
Tmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Í
Tmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*I
value@B> B8x (model/lambda/map/while/random_flip_up_down/Rank:0) = ß
Tmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*[
valueRBP BJy (model/lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = ¥
Mmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssertLmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/All:output:0]model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0]model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0]model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:08model/lambda/map/while/random_flip_up_down/Rank:output:0]model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0Jmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0U^model/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 »
=model/lambda/map/while/random_flip_up_down/control_dependencyIdentity?model/lambda/map/while/random_flip_left_right/Identity:output:0N^model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertU^model/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*I
_class?
=;loc:@model/lambda/map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
?model/lambda/map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
=model/lambda/map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=model/lambda/map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ð
Gmodel/lambda/map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniformHmodel/lambda/map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0ÿ
=model/lambda/map/while/random_flip_up_down/random_uniform/MulMulPmodel/lambda/map/while/random_flip_up_down/random_uniform/RandomUniform:output:0Fmodel/lambda/map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: v
1model/lambda/map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?×
/model/lambda/map/while/random_flip_up_down/LessLessAmodel/lambda/map/while/random_flip_up_down/random_uniform/Mul:z:0:model/lambda/map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: ¥
*model/lambda/map/while/random_flip_up_downStatelessIf3model/lambda/map/while/random_flip_up_down/Less:z:0Fmodel/lambda/map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_lambda_map_while_random_flip_up_down_false_154308*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*I
then_branch:R8
6model_lambda_map_while_random_flip_up_down_true_154307³
3model/lambda/map/while/random_flip_up_down/IdentityIdentity3model/lambda/map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
5model/lambda/map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB x
3model/lambda/map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½x
3model/lambda/map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¢
Qmodel/lambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      ÷
Lmodel/lambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterZmodel/lambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Lmodel/lambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :¿
Hmodel/lambda/map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2>model/lambda/map/while/stateless_random_uniform/shape:output:0Rmodel/lambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Vmodel/lambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Umodel/lambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ×
3model/lambda/map/while/stateless_random_uniform/subSub<model/lambda/map/while/stateless_random_uniform/max:output:0<model/lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ç
3model/lambda/map/while/stateless_random_uniform/mulMulQmodel/lambda/map/while/stateless_random_uniform/StatelessRandomUniformV2:output:07model/lambda/map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: Ð
/model/lambda/map/while/stateless_random_uniformAddV27model/lambda/map/while/stateless_random_uniform/mul:z:0<model/lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ã
(model/lambda/map/while/adjust_brightnessAddV2<model/lambda/map/while/random_flip_up_down/Identity:output:03model/lambda/map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
1model/lambda/map/while/adjust_brightness/IdentityIdentity,model/lambda/map/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
+model/lambda/map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB n
)model/lambda/map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @n
)model/lambda/map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @¨
3model/lambda/map/while/random_uniform/RandomUniformRandomUniform4model/lambda/map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0¹
)model/lambda/map/while/random_uniform/subSub2model/lambda/map/while/random_uniform/max:output:02model/lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: ¾
)model/lambda/map/while/random_uniform/mulMul<model/lambda/map/while/random_uniform/RandomUniform:output:0-model/lambda/map/while/random_uniform/sub:z:0*
T0*
_output_shapes
: ²
%model/lambda/map/while/random_uniformAddV2-model/lambda/map/while/random_uniform/mul:z:02model/lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: ê
9model/lambda/map/while/adjust_saturation/AdjustSaturationAdjustSaturation:model/lambda/map/while/adjust_brightness/Identity:output:0)model/lambda/map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
1model/lambda/map/while/adjust_saturation/IdentityIdentityBmodel/lambda/map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)model/lambda/map/while/central_crop/ShapeShape:model/lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:
7model/lambda/map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
9model/lambda/map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
9model/lambda/map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1model/lambda/map/while/central_crop/strided_sliceStridedSlice2model/lambda/map/while/central_crop/Shape:output:0@model/lambda/map/while/central_crop/strided_slice/stack:output:0Bmodel/lambda/map/while/central_crop/strided_slice/stack_1:output:0Bmodel/lambda/map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask{
9model/lambda/map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ñ
Dmodel/lambda/map/while/central_crop/assert_positive/assert_less/LessLessBmodel/lambda/map/while/central_crop/assert_positive/Const:output:0:model/lambda/map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
Emodel/lambda/map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ü
Cmodel/lambda/map/while/central_crop/assert_positive/assert_less/AllAllHmodel/lambda/map/while/central_crop/assert_positive/assert_less/Less:z:0Nmodel/lambda/map/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ·
Lmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¿
Tmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.ö
Mmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertAssertLmodel/lambda/map/while/central_crop/assert_positive/assert_less/All:output:0]model/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0N^model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 j
(model/lambda/map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :|
:model/lambda/map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :î
Emodel/lambda/map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual1model/lambda/map/while/central_crop/Rank:output:0Cmodel/lambda/map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
=model/lambda/map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Dmodel/lambda/map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Dmodel/lambda/map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ç
>model/lambda/map/while/central_crop/assert_greater_equal/rangeRangeMmodel/lambda/map/while/central_crop/assert_greater_equal/range/start:output:0Fmodel/lambda/map/while/central_crop/assert_greater_equal/Rank:output:0Mmodel/lambda/map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: ï
<model/lambda/map/while/central_crop/assert_greater_equal/AllAllImodel/lambda/map/while/central_crop/assert_greater_equal/GreaterEqual:z:0Gmodel/lambda/map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: ±
Emodel/lambda/map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Gmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¹
Gmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*B
value9B7 B1x (model/lambda/map/while/central_crop/Rank:0) = Ë
Gmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*T
valueKBI BCy (model/lambda/map/while/central_crop/assert_greater_equal/y:0) = ¹
Mmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.¹
Mmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:¿
Mmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*B
value9B7 B1x (model/lambda/map/while/central_crop/Rank:0) = Ñ
Mmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*T
valueKBI BCy (model/lambda/map/while/central_crop/assert_greater_equal/y:0) = æ
Fmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/AssertAssertEmodel/lambda/map/while/central_crop/assert_greater_equal/All:output:0Vmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Vmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Vmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:01model/lambda/map/while/central_crop/Rank:output:0Vmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:0Cmodel/lambda/map/while/central_crop/assert_greater_equal/y:output:0N^model/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 
6model/lambda/map/while/central_crop/control_dependencyIdentity:model/lambda/map/while/adjust_saturation/Identity:output:0G^model/lambda/map/while/central_crop/assert_greater_equal/Assert/AssertN^model/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*D
_class:
86loc:@model/lambda/map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+model/lambda/map/while/central_crop/Shape_1Shape:model/lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:
9model/lambda/map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model/lambda/map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model/lambda/map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model/lambda/map/while/central_crop/strided_slice_1StridedSlice4model/lambda/map/while/central_crop/Shape_1:output:0Bmodel/lambda/map/while/central_crop/strided_slice_1/stack:output:0Dmodel/lambda/map/while/central_crop/strided_slice_1/stack_1:output:0Dmodel/lambda/map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
+model/lambda/map/while/central_crop/Shape_2Shape:model/lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:
9model/lambda/map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
;model/lambda/map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model/lambda/map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model/lambda/map/while/central_crop/strided_slice_2StridedSlice4model/lambda/map/while/central_crop/Shape_2:output:0Bmodel/lambda/map/while/central_crop/strided_slice_2/stack:output:0Dmodel/lambda/map/while/central_crop/strided_slice_2/stack_1:output:0Dmodel/lambda/map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(model/lambda/map/while/central_crop/CastCast<model/lambda/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,model/lambda/map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *üT;?
*model/lambda/map/while/central_crop/Cast_1Cast5model/lambda/map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ­
'model/lambda/map/while/central_crop/mulMul,model/lambda/map/while/central_crop/Cast:y:0.model/lambda/map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: ª
'model/lambda/map/while/central_crop/subSub,model/lambda/map/while/central_crop/Cast:y:0+model/lambda/map/while/central_crop/mul:z:0*
T0*
_output_shapes
: v
-model/lambda/map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @¼
+model/lambda/map/while/central_crop/truedivRealDiv+model/lambda/map/while/central_crop/sub:z:06model/lambda/map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: 
*model/lambda/map/while/central_crop/Cast_2Cast/model/lambda/map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:  
*model/lambda/map/while/central_crop/Cast_3Cast<model/lambda/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: q
,model/lambda/map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *üT;?
*model/lambda/map/while/central_crop/Cast_4Cast5model/lambda/map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ±
)model/lambda/map/while/central_crop/mul_1Mul.model/lambda/map/while/central_crop/Cast_3:y:0.model/lambda/map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: °
)model/lambda/map/while/central_crop/sub_1Sub.model/lambda/map/while/central_crop/Cast_3:y:0-model/lambda/map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: x
/model/lambda/map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @Â
-model/lambda/map/while/central_crop/truediv_1RealDiv-model/lambda/map/while/central_crop/sub_1:z:08model/lambda/map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: 
*model/lambda/map/while/central_crop/Cast_5Cast1model/lambda/map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: m
+model/lambda/map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :·
)model/lambda/map/while/central_crop/mul_2Mul.model/lambda/map/while/central_crop/Cast_2:y:04model/lambda/map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: ¾
)model/lambda/map/while/central_crop/sub_2Sub<model/lambda/map/while/central_crop/strided_slice_1:output:0-model/lambda/map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: m
+model/lambda/map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :·
)model/lambda/map/while/central_crop/mul_3Mul.model/lambda/map/while/central_crop/Cast_5:y:04model/lambda/map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: ¾
)model/lambda/map/while/central_crop/sub_3Sub<model/lambda/map/while/central_crop/strided_slice_2:output:0-model/lambda/map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: m
+model/lambda/map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : õ
)model/lambda/map/while/central_crop/stackPack.model/lambda/map/while/central_crop/Cast_2:y:0.model/lambda/map/while/central_crop/Cast_5:y:04model/lambda/map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:x
-model/lambda/map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ÷
+model/lambda/map/while/central_crop/stack_1Pack-model/lambda/map/while/central_crop/sub_2:z:0-model/lambda/map/while/central_crop/sub_3:z:06model/lambda/map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:¤
)model/lambda/map/while/central_crop/SliceSlice:model/lambda/map/while/adjust_saturation/Identity:output:02model/lambda/map/while/central_crop/stack:output:04model/lambda/map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿn
,model/lambda/map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ä
(model/lambda/map/while/resize/ExpandDims
ExpandDims2model/lambda/map/while/central_crop/Slice:output:05model/lambda/map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿs
"model/lambda/map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   ë
,model/lambda/map/while/resize/ResizeBilinearResizeBilinear1model/lambda/map/while/resize/ExpandDims:output:0+model/lambda/map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(µ
%model/lambda/map/while/resize/SqueezeSqueeze=model/lambda/map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 
;model/lambda/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$model_lambda_map_while_placeholder_1"model_lambda_map_while_placeholder.model/lambda/map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒ^
model/lambda/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model/lambda/map/while/addAddV2"model_lambda_map_while_placeholder%model/lambda/map/while/add/y:output:0*
T0*
_output_shapes
: `
model/lambda/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :«
model/lambda/map/while/add_1AddV2:model_lambda_map_while_model_lambda_map_while_loop_counter'model/lambda/map/while/add_1/y:output:0*
T0*
_output_shapes
: 
model/lambda/map/while/IdentityIdentity model/lambda/map/while/add_1:z:0^model/lambda/map/while/NoOp*
T0*
_output_shapes
: £
!model/lambda/map/while/Identity_1Identity5model_lambda_map_while_model_lambda_map_strided_slice^model/lambda/map/while/NoOp*
T0*
_output_shapes
: 
!model/lambda/map/while/Identity_2Identitymodel/lambda/map/while/add:z:0^model/lambda/map/while/NoOp*
T0*
_output_shapes
: Ì
!model/lambda/map/while/Identity_3IdentityKmodel/lambda/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/lambda/map/while/NoOp*
T0*
_output_shapes
: :éèÒÊ
model/lambda/map/while/NoOpNoOpG^model/lambda/map/while/central_crop/assert_greater_equal/Assert/AssertN^model/lambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertQ^model/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertX^model/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertN^model/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertU^model/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "K
model_lambda_map_while_identity(model/lambda/map/while/Identity:output:0"O
!model_lambda_map_while_identity_1*model/lambda/map/while/Identity_1:output:0"O
!model_lambda_map_while_identity_2*model/lambda/map/while/Identity_2:output:0"O
!model_lambda_map_while_identity_3*model/lambda/map/while/Identity_3:output:0"t
7model_lambda_map_while_model_lambda_map_strided_slice_19model_lambda_map_while_model_lambda_map_strided_slice_1_0"ì
smodel_lambda_map_while_tensorarrayv2read_tensorlistgetitem_model_lambda_map_tensorarrayunstack_tensorlistfromtensorumodel_lambda_map_while_tensorarrayv2read_tensorlistgetitem_model_lambda_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2
Fmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/AssertFmodel/lambda/map/while/central_crop/assert_greater_equal/Assert/Assert2
Mmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertMmodel/lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert2¤
Pmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertPmodel/lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert2²
Wmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertWmodel/lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
Mmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertMmodel/lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2¬
Tmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertTmodel/lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ë
²
&__inference_vgg16_layer_call_fn_158839

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_155165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

c
G__inference_block2_pool_layer_call_and_return_conditional_losses_159163

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
âå
!
!__inference__wrapped_model_154538	
bytes\
Bmodel_sequential_vgg16_block1_conv1_conv2d_readvariableop_resource:@Q
Cmodel_sequential_vgg16_block1_conv1_biasadd_readvariableop_resource:@\
Bmodel_sequential_vgg16_block1_conv2_conv2d_readvariableop_resource:@@Q
Cmodel_sequential_vgg16_block1_conv2_biasadd_readvariableop_resource:@]
Bmodel_sequential_vgg16_block2_conv1_conv2d_readvariableop_resource:@R
Cmodel_sequential_vgg16_block2_conv1_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block2_conv2_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block2_conv2_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block3_conv1_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block3_conv1_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block3_conv2_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block3_conv2_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block3_conv3_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block3_conv3_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block4_conv1_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block4_conv1_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block4_conv2_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block4_conv2_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block4_conv3_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block4_conv3_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block5_conv1_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block5_conv1_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block5_conv2_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block5_conv2_biasadd_readvariableop_resource:	^
Bmodel_sequential_vgg16_block5_conv3_conv2d_readvariableop_resource:R
Cmodel_sequential_vgg16_block5_conv3_biasadd_readvariableop_resource:	H
5model_sequential_dense_matmul_readvariableop_resource:	D
6model_sequential_dense_biasadd_readvariableop_resource:
identity	

identity_1¢model/lambda/map/while¢-model/sequential/dense/BiasAdd/ReadVariableOp¢,model/sequential/dense/MatMul/ReadVariableOp¢:model/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp¢:model/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp¢9model/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpK
model/lambda/map/ShapeShapebytes*
T0*
_output_shapes
:n
$model/lambda/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model/lambda/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model/lambda/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
model/lambda/map/strided_sliceStridedSlicemodel/lambda/map/Shape:output:0-model/lambda/map/strided_slice/stack:output:0/model/lambda/map/strided_slice/stack_1:output:0/model/lambda/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,model/lambda/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿå
model/lambda/map/TensorArrayV2TensorListReserve5model/lambda/map/TensorArrayV2/element_shape:output:0'model/lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖl
)model/lambda/map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ý
8model/lambda/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorbytes2model/lambda/map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖX
model/lambda/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : y
.model/lambda/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿé
 model/lambda/map/TensorArrayV2_1TensorListReserve7model/lambda/map/TensorArrayV2_1/element_shape:output:0'model/lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
#model/lambda/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : æ
model/lambda/map/whileWhile,model/lambda/map/while/loop_counter:output:0'model/lambda/map/strided_slice:output:0model/lambda/map/Const:output:0)model/lambda/map/TensorArrayV2_1:handle:0'model/lambda/map/strided_slice:output:0Hmodel/lambda/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *.
body&R$
"model_lambda_map_while_body_154200*.
cond&R$
"model_lambda_map_while_cond_154199*
output_shapes
: : : : : : 
Amodel/lambda/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      û
3model/lambda/map/TensorArrayV2Stack/TensorListStackTensorListStackmodel/lambda/map/while:output:3Jmodel/lambda/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0Ä
9model/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
*model/sequential/vgg16/block1_conv1/Conv2DConv2D<model/lambda/map/TensorArrayV2Stack/TensorListStack:tensor:0Amodel/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
º
:model/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ë
+model/sequential/vgg16/block1_conv1/BiasAddBiasAdd3model/sequential/vgg16/block1_conv1/Conv2D:output:0Bmodel/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(model/sequential/vgg16/block1_conv1/ReluRelu4model/sequential/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Ä
9model/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
*model/sequential/vgg16/block1_conv2/Conv2DConv2D6model/sequential/vgg16/block1_conv1/Relu:activations:0Amodel/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
º
:model/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ë
+model/sequential/vgg16/block1_conv2/BiasAddBiasAdd3model/sequential/vgg16/block1_conv2/Conv2D:output:0Bmodel/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¢
(model/sequential/vgg16/block1_conv2/ReluRelu4model/sequential/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Ú
*model/sequential/vgg16/block1_pool/MaxPoolMaxPool6model/sequential/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
Å
9model/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
*model/sequential/vgg16/block2_conv1/Conv2DConv2D3model/sequential/vgg16/block1_pool/MaxPool:output:0Amodel/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
»
:model/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block2_conv1/BiasAddBiasAdd3model/sequential/vgg16/block2_conv1/Conv2D:output:0Bmodel/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¡
(model/sequential/vgg16/block2_conv1/ReluRelu4model/sequential/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÆ
9model/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block2_conv2/Conv2DConv2D6model/sequential/vgg16/block2_conv1/Relu:activations:0Amodel/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
»
:model/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block2_conv2/BiasAddBiasAdd3model/sequential/vgg16/block2_conv2/Conv2D:output:0Bmodel/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¡
(model/sequential/vgg16/block2_conv2/ReluRelu4model/sequential/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÛ
*model/sequential/vgg16/block2_pool/MaxPoolMaxPool6model/sequential/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
Æ
9model/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block3_conv1/Conv2DConv2D3model/sequential/vgg16/block2_pool/MaxPool:output:0Amodel/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
»
:model/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block3_conv1/BiasAddBiasAdd3model/sequential/vgg16/block3_conv1/Conv2D:output:0Bmodel/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
(model/sequential/vgg16/block3_conv1/ReluRelu4model/sequential/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Æ
9model/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block3_conv2/Conv2DConv2D6model/sequential/vgg16/block3_conv1/Relu:activations:0Amodel/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
»
:model/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block3_conv2/BiasAddBiasAdd3model/sequential/vgg16/block3_conv2/Conv2D:output:0Bmodel/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
(model/sequential/vgg16/block3_conv2/ReluRelu4model/sequential/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Æ
9model/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block3_conv3/Conv2DConv2D6model/sequential/vgg16/block3_conv2/Relu:activations:0Amodel/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
»
:model/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block3_conv3/BiasAddBiasAdd3model/sequential/vgg16/block3_conv3/Conv2D:output:0Bmodel/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¡
(model/sequential/vgg16/block3_conv3/ReluRelu4model/sequential/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Û
*model/sequential/vgg16/block3_pool/MaxPoolMaxPool6model/sequential/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Æ
9model/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block4_conv1/Conv2DConv2D3model/sequential/vgg16/block3_pool/MaxPool:output:0Amodel/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block4_conv1/BiasAddBiasAdd3model/sequential/vgg16/block4_conv1/Conv2D:output:0Bmodel/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block4_conv1/ReluRelu4model/sequential/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
9model/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block4_conv2/Conv2DConv2D6model/sequential/vgg16/block4_conv1/Relu:activations:0Amodel/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block4_conv2/BiasAddBiasAdd3model/sequential/vgg16/block4_conv2/Conv2D:output:0Bmodel/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block4_conv2/ReluRelu4model/sequential/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
9model/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block4_conv3/Conv2DConv2D6model/sequential/vgg16/block4_conv2/Relu:activations:0Amodel/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block4_conv3/BiasAddBiasAdd3model/sequential/vgg16/block4_conv3/Conv2D:output:0Bmodel/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block4_conv3/ReluRelu4model/sequential/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
*model/sequential/vgg16/block4_pool/MaxPoolMaxPool6model/sequential/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Æ
9model/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block5_conv1/Conv2DConv2D3model/sequential/vgg16/block4_pool/MaxPool:output:0Amodel/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block5_conv1/BiasAddBiasAdd3model/sequential/vgg16/block5_conv1/Conv2D:output:0Bmodel/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block5_conv1/ReluRelu4model/sequential/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
9model/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block5_conv2/Conv2DConv2D6model/sequential/vgg16/block5_conv1/Relu:activations:0Amodel/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block5_conv2/BiasAddBiasAdd3model/sequential/vgg16/block5_conv2/Conv2D:output:0Bmodel/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block5_conv2/ReluRelu4model/sequential/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
9model/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_sequential_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
*model/sequential/vgg16/block5_conv3/Conv2DConv2D6model/sequential/vgg16/block5_conv2/Relu:activations:0Amodel/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
:model/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpCmodel_sequential_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ê
+model/sequential/vgg16/block5_conv3/BiasAddBiasAdd3model/sequential/vgg16/block5_conv3/Conv2D:output:0Bmodel/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
(model/sequential/vgg16/block5_conv3/ReluRelu4model/sequential/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
*model/sequential/vgg16/block5_pool/MaxPoolMaxPool6model/sequential/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

Fmodel/sequential/vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      õ
4model/sequential/vgg16/global_average_pooling2d/MeanMean3model/sequential/vgg16/block5_pool/MaxPool:output:0Omodel/sequential/vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,model/sequential/dense/MatMul/ReadVariableOpReadVariableOp5model_sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Î
model/sequential/dense/MatMulMatMul=model/sequential/vgg16/global_average_pooling2d/Mean:output:04model/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp6model_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
model/sequential/dense/BiasAddBiasAdd'model/sequential/dense/MatMul:product:05model/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/sequential/dense/SoftmaxSoftmax'model/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
model/CLASSES/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
model/CLASSES/ArgMaxArgMax(model/sequential/dense/Softmax:softmax:0'model/CLASSES/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitymodel/CLASSES/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(model/sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp^model/lambda/map/while.^model/sequential/dense/BiasAdd/ReadVariableOp-^model/sequential/dense/MatMul/ReadVariableOp;^model/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp;^model/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp;^model/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp;^model/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp;^model/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp;^model/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp;^model/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp;^model/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp;^model/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp;^model/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp;^model/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp;^model/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp;^model/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp:^model/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 20
model/lambda/map/whilemodel/lambda/map/while2^
-model/sequential/dense/BiasAdd/ReadVariableOp-model/sequential/dense/BiasAdd/ReadVariableOp2\
,model/sequential/dense/MatMul/ReadVariableOp,model/sequential/dense/MatMul/ReadVariableOp2x
:model/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp:model/sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp9model/sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp:model/sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp9model/sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp:model/sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp9model/sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp:model/sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp9model/sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp:model/sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp9model/sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp:model/sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp9model/sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp:model/sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp9model/sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp:model/sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp9model/sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp:model/sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp9model/sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp:model/sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp9model/sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp:model/sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp9model/sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp:model/sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp9model/sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp2x
:model/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp:model/sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp2v
9model/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp9model/sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes
´
Õ
lambda_map_while_body_1571472
.lambda_map_while_lambda_map_while_loop_counter-
)lambda_map_while_lambda_map_strided_slice 
lambda_map_while_placeholder"
lambda_map_while_placeholder_11
-lambda_map_while_lambda_map_strided_slice_1_0m
ilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0
lambda_map_while_identity
lambda_map_while_identity_1
lambda_map_while_identity_2
lambda_map_while_identity_3/
+lambda_map_while_lambda_map_strided_slice_1k
glambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor¢@lambda/map/while/central_crop/assert_greater_equal/Assert/Assert¢Glambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert¢Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert¢Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert¢Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assertk
(lambda/map/while/TensorArrayV2Read/ConstConst*
_output_shapes
: *
dtype0*
valueB ²
4lambda/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0lambda_map_while_placeholder1lambda/map/while/TensorArrayV2Read/Const:output:0*
_output_shapes
: *
element_dtype0¬
lambda/map/while/DecodeJpeg
DecodeJpeg;lambda/map/while/TensorArrayV2Read/TensorListGetItem:item:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
lambda/map/while/CastCast#lambda/map/while/DecodeJpeg:image:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ_
lambda/map/while/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C¢
lambda/map/while/truedivRealDivlambda/map/while/Cast:y:0#lambda/map/while/truediv/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
-lambda/map/while/random_flip_left_right/ShapeShapelambda/map/while/truediv:z:0*
T0*
_output_shapes
:
;lambda/map/while/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
=lambda/map/while/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=lambda/map/while/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5lambda/map/while/random_flip_left_right/strided_sliceStridedSlice6lambda/map/while/random_flip_left_right/Shape:output:0Dlambda/map/while/random_flip_left_right/strided_slice/stack:output:0Flambda/map/while/random_flip_left_right/strided_slice/stack_1:output:0Flambda/map/while/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
=lambda/map/while/random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ý
Hlambda/map/while/random_flip_left_right/assert_positive/assert_less/LessLessFlambda/map/while/random_flip_left_right/assert_positive/Const:output:0>lambda/map/while/random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:
Ilambda/map/while/random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Glambda/map/while/random_flip_left_right/assert_positive/assert_less/AllAllLlambda/map/while/random_flip_left_right/assert_positive/assert_less/Less:z:0Rlambda/map/while/random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: »
Plambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Ã
Xlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.²
Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertPlambda/map/while/random_flip_left_right/assert_positive/assert_less/All:output:0alambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 n
,lambda/map/while/random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :
>lambda/map/while/random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :ú
Ilambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual5lambda/map/while/random_flip_left_right/Rank:output:0Glambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
Alambda/map/while/random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Hlambda/map/while/random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Hlambda/map/while/random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :×
Blambda/map/while/random_flip_left_right/assert_greater_equal/rangeRangeQlambda/map/while/random_flip_left_right/assert_greater_equal/range/start:output:0Jlambda/map/while/random_flip_left_right/assert_greater_equal/Rank:output:0Qlambda/map/while/random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: û
@lambda/map/while/random_flip_left_right/assert_greater_equal/AllAllMlambda/map/while/random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Klambda/map/while/random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: µ
Ilambda/map/while/random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.·
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Á
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*F
value=B; B5x (lambda/map/while/random_flip_left_right/Rank:0) = Ó
Klambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*X
valueOBM BGy (lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = ½
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.½
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Ç
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*F
value=B; B5x (lambda/map/while/random_flip_left_right/Rank:0) = Ù
Qlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*X
valueOBM BGy (lambda/map/while/random_flip_left_right/assert_greater_equal/y:0) = 
Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertAssertIlambda/map/while/random_flip_left_right/assert_greater_equal/All:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:05lambda/map/while/random_flip_left_right/Rank:output:0Zlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0Glambda/map/while/random_flip_left_right/assert_greater_equal/y:output:0R^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 ñ
:lambda/map/while/random_flip_left_right/control_dependencyIdentitylambda/map/while/truediv:z:0K^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertR^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*+
_class!
loc:@lambda/map/while/truediv*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
<lambda/map/while/random_flip_left_right/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
:lambda/map/while/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:lambda/map/while/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ê
Dlambda/map/while/random_flip_left_right/random_uniform/RandomUniformRandomUniformElambda/map/while/random_flip_left_right/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0ö
:lambda/map/while/random_flip_left_right/random_uniform/MulMulMlambda/map/while/random_flip_left_right/random_uniform/RandomUniform:output:0Clambda/map/while/random_flip_left_right/random_uniform/max:output:0*
T0*
_output_shapes
: s
.lambda/map/while/random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Î
,lambda/map/while/random_flip_left_right/LessLess>lambda/map/while/random_flip_left_right/random_uniform/Mul:z:07lambda/map/while/random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 
'lambda/map/while/random_flip_left_rightStatelessIf0lambda/map/while/random_flip_left_right/Less:z:0Clambda/map/while/random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *G
else_branch8R6
4lambda_map_while_random_flip_left_right_false_157208*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*F
then_branch7R5
3lambda_map_while_random_flip_left_right_true_157207­
0lambda/map/while/random_flip_left_right/IdentityIdentity0lambda/map/while/random_flip_left_right:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*lambda/map/while/random_flip_up_down/ShapeShape9lambda/map/while/random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:
8lambda/map/while/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:lambda/map/while/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:lambda/map/while/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2lambda/map/while/random_flip_up_down/strided_sliceStridedSlice3lambda/map/while/random_flip_up_down/Shape:output:0Alambda/map/while/random_flip_up_down/strided_slice/stack:output:0Clambda/map/while/random_flip_up_down/strided_slice/stack_1:output:0Clambda/map/while/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask|
:lambda/map/while/random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ô
Elambda/map/while/random_flip_up_down/assert_positive/assert_less/LessLessClambda/map/while/random_flip_up_down/assert_positive/Const:output:0;lambda/map/while/random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:
Flambda/map/while/random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ÿ
Dlambda/map/while/random_flip_up_down/assert_positive/assert_less/AllAllIlambda/map/while/random_flip_up_down/assert_positive/assert_less/Less:z:0Olambda/map/while/random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: ¸
Mlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.À
Ulambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.ö
Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertMlambda/map/while/random_flip_up_down/assert_positive/assert_less/All:output:0^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0K^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 k
)lambda/map/while/random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;lambda/map/while/random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :ñ
Flambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual2lambda/map/while/random_flip_up_down/Rank:output:0Dlambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 
>lambda/map/while/random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
Elambda/map/while/random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
Elambda/map/while/random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ë
?lambda/map/while/random_flip_up_down/assert_greater_equal/rangeRangeNlambda/map/while/random_flip_up_down/assert_greater_equal/range/start:output:0Glambda/map/while/random_flip_up_down/assert_greater_equal/Rank:output:0Nlambda/map/while/random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: ò
=lambda/map/while/random_flip_up_down/assert_greater_equal/AllAllJlambda/map/while/random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Hlambda/map/while/random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: ²
Flambda/map/while/random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.´
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:»
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2x (lambda/map/while/random_flip_up_down/Rank:0) = Í
Hlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = º
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.º
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:Á
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*C
value:B8 B2x (lambda/map/while/random_flip_up_down/Rank:0) = Ó
Nlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*U
valueLBJ BDy (lambda/map/while/random_flip_up_down/assert_greater_equal/y:0) = ï
Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertAssertFlambda/map/while/random_flip_up_down/assert_greater_equal/All:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:02lambda/map/while/random_flip_up_down/Rank:output:0Wlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0Dlambda/map/while/random_flip_up_down/assert_greater_equal/y:output:0O^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 
7lambda/map/while/random_flip_up_down/control_dependencyIdentity9lambda/map/while/random_flip_left_right/Identity:output:0H^lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertO^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*C
_class9
75loc:@lambda/map/while/random_flip_left_right/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
9lambda/map/while/random_flip_up_down/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB |
7lambda/map/while/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    |
7lambda/map/while/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
Alambda/map/while/random_flip_up_down/random_uniform/RandomUniformRandomUniformBlambda/map/while/random_flip_up_down/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0í
7lambda/map/while/random_flip_up_down/random_uniform/MulMulJlambda/map/while/random_flip_up_down/random_uniform/RandomUniform:output:0@lambda/map/while/random_flip_up_down/random_uniform/max:output:0*
T0*
_output_shapes
: p
+lambda/map/while/random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
)lambda/map/while/random_flip_up_down/LessLess;lambda/map/while/random_flip_up_down/random_uniform/Mul:z:04lambda/map/while/random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 
$lambda/map/while/random_flip_up_downStatelessIf-lambda/map/while/random_flip_up_down/Less:z:0@lambda/map/while/random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *D
else_branch5R3
1lambda_map_while_random_flip_up_down_false_157255*3
output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*C
then_branch4R2
0lambda_map_while_random_flip_up_down_true_157254§
-lambda/map/while/random_flip_up_down/IdentityIdentity-lambda/map/while/random_flip_up_down:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
/lambda/map/while/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB r
-lambda/map/while/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ½r
-lambda/map/while/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=
Klambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB"      ë
Flambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterTlambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Flambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :¡
Blambda/map/while/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV28lambda/map/while/stateless_random_uniform/shape:output:0Llambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Plambda/map/while/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Olambda/map/while/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: Å
-lambda/map/while/stateless_random_uniform/subSub6lambda/map/while/stateless_random_uniform/max:output:06lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Õ
-lambda/map/while/stateless_random_uniform/mulMulKlambda/map/while/stateless_random_uniform/StatelessRandomUniformV2:output:01lambda/map/while/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ¾
)lambda/map/while/stateless_random_uniformAddV21lambda/map/while/stateless_random_uniform/mul:z:06lambda/map/while/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ñ
"lambda/map/while/adjust_brightnessAddV26lambda/map/while/random_flip_up_down/Identity:output:0-lambda/map/while/stateless_random_uniform:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+lambda/map/while/adjust_brightness/IdentityIdentity&lambda/map/while/adjust_brightness:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
%lambda/map/while/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB h
#lambda/map/while/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
#lambda/map/while/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  @
-lambda/map/while/random_uniform/RandomUniformRandomUniform.lambda/map/while/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype0§
#lambda/map/while/random_uniform/subSub,lambda/map/while/random_uniform/max:output:0,lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: ¬
#lambda/map/while/random_uniform/mulMul6lambda/map/while/random_uniform/RandomUniform:output:0'lambda/map/while/random_uniform/sub:z:0*
T0*
_output_shapes
:  
lambda/map/while/random_uniformAddV2'lambda/map/while/random_uniform/mul:z:0,lambda/map/while/random_uniform/min:output:0*
T0*
_output_shapes
: Ø
3lambda/map/while/adjust_saturation/AdjustSaturationAdjustSaturation4lambda/map/while/adjust_brightness/Identity:output:0#lambda/map/while/random_uniform:z:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
+lambda/map/while/adjust_saturation/IdentityIdentity<lambda/map/while/adjust_saturation/AdjustSaturation:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
#lambda/map/while/central_crop/ShapeShape4lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:
1lambda/map/while/central_crop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
3lambda/map/while/central_crop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3lambda/map/while/central_crop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
+lambda/map/while/central_crop/strided_sliceStridedSlice,lambda/map/while/central_crop/Shape:output:0:lambda/map/while/central_crop/strided_slice/stack:output:0<lambda/map/while/central_crop/strided_slice/stack_1:output:0<lambda/map/while/central_crop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_masku
3lambda/map/while/central_crop/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : ß
>lambda/map/while/central_crop/assert_positive/assert_less/LessLess<lambda/map/while/central_crop/assert_positive/Const:output:04lambda/map/while/central_crop/strided_slice:output:0*
T0*
_output_shapes
:
?lambda/map/while/central_crop/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: ê
=lambda/map/while/central_crop/assert_positive/assert_less/AllAllBlambda/map/while/central_crop/assert_positive/assert_less/Less:z:0Hlambda/map/while/central_crop/assert_positive/assert_less/Const:output:0*
_output_shapes
: ±
Flambda/map/while/central_crop/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.¹
Nlambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.Þ
Glambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertAssertFlambda/map/while/central_crop/assert_positive/assert_less/All:output:0Wlambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert/data_0:output:0H^lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert*

T
2*
_output_shapes
 d
"lambda/map/while/central_crop/RankConst*
_output_shapes
: *
dtype0*
value	B :v
4lambda/map/while/central_crop/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :Ü
?lambda/map/while/central_crop/assert_greater_equal/GreaterEqualGreaterEqual+lambda/map/while/central_crop/Rank:output:0=lambda/map/while/central_crop/assert_greater_equal/y:output:0*
T0*
_output_shapes
: y
7lambda/map/while/central_crop/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 
>lambda/map/while/central_crop/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
>lambda/map/while/central_crop/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¯
8lambda/map/while/central_crop/assert_greater_equal/rangeRangeGlambda/map/while/central_crop/assert_greater_equal/range/start:output:0@lambda/map/while/central_crop/assert_greater_equal/Rank:output:0Glambda/map/while/central_crop/assert_greater_equal/range/delta:output:0*
_output_shapes
: Ý
6lambda/map/while/central_crop/assert_greater_equal/AllAllClambda/map/while/central_crop/assert_greater_equal/GreaterEqual:z:0Alambda/map/while/central_crop/assert_greater_equal/range:output:0*
_output_shapes
: «
?lambda/map/while/central_crop/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.­
Alambda/map/while/central_crop/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:­
Alambda/map/while/central_crop/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (lambda/map/while/central_crop/Rank:0) = ¿
Alambda/map/while/central_crop/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (lambda/map/while/central_crop/assert_greater_equal/y:0) = ³
Glambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.³
Glambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:³
Glambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (lambda/map/while/central_crop/Rank:0) = Å
Glambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (lambda/map/while/central_crop/assert_greater_equal/y:0) = °
@lambda/map/while/central_crop/assert_greater_equal/Assert/AssertAssert?lambda/map/while/central_crop/assert_greater_equal/All:output:0Plambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_0:output:0Plambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_1:output:0Plambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_2:output:0+lambda/map/while/central_crop/Rank:output:0Plambda/map/while/central_crop/assert_greater_equal/Assert/Assert/data_4:output:0=lambda/map/while/central_crop/assert_greater_equal/y:output:0H^lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 þ
0lambda/map/while/central_crop/control_dependencyIdentity4lambda/map/while/adjust_saturation/Identity:output:0A^lambda/map/while/central_crop/assert_greater_equal/Assert/AssertH^lambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert*
T0*>
_class4
20loc:@lambda/map/while/adjust_saturation/Identity*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
%lambda/map/while/central_crop/Shape_1Shape4lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
3lambda/map/while/central_crop/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5lambda/map/while/central_crop/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5lambda/map/while/central_crop/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-lambda/map/while/central_crop/strided_slice_1StridedSlice.lambda/map/while/central_crop/Shape_1:output:0<lambda/map/while/central_crop/strided_slice_1/stack:output:0>lambda/map/while/central_crop/strided_slice_1/stack_1:output:0>lambda/map/while/central_crop/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%lambda/map/while/central_crop/Shape_2Shape4lambda/map/while/adjust_saturation/Identity:output:0*
T0*
_output_shapes
:}
3lambda/map/while/central_crop/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
5lambda/map/while/central_crop/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5lambda/map/while/central_crop/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-lambda/map/while/central_crop/strided_slice_2StridedSlice.lambda/map/while/central_crop/Shape_2:output:0<lambda/map/while/central_crop/strided_slice_2/stack:output:0>lambda/map/while/central_crop/strided_slice_2/stack_1:output:0>lambda/map/while/central_crop/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
"lambda/map/while/central_crop/CastCast6lambda/map/while/central_crop/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: k
&lambda/map/while/central_crop/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *^@?
$lambda/map/while/central_crop/Cast_1Cast/lambda/map/while/central_crop/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!lambda/map/while/central_crop/mulMul&lambda/map/while/central_crop/Cast:y:0(lambda/map/while/central_crop/Cast_1:y:0*
T0*
_output_shapes
: 
!lambda/map/while/central_crop/subSub&lambda/map/while/central_crop/Cast:y:0%lambda/map/while/central_crop/mul:z:0*
T0*
_output_shapes
: p
'lambda/map/while/central_crop/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2       @ª
%lambda/map/while/central_crop/truedivRealDiv%lambda/map/while/central_crop/sub:z:00lambda/map/while/central_crop/truediv/y:output:0*
T0*
_output_shapes
: 
$lambda/map/while/central_crop/Cast_2Cast)lambda/map/while/central_crop/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: 
$lambda/map/while/central_crop/Cast_3Cast6lambda/map/while/central_crop/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: k
&lambda/map/while/central_crop/Cast_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *^@?
$lambda/map/while/central_crop/Cast_4Cast/lambda/map/while/central_crop/Cast_4/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
#lambda/map/while/central_crop/mul_1Mul(lambda/map/while/central_crop/Cast_3:y:0(lambda/map/while/central_crop/Cast_4:y:0*
T0*
_output_shapes
: 
#lambda/map/while/central_crop/sub_1Sub(lambda/map/while/central_crop/Cast_3:y:0'lambda/map/while/central_crop/mul_1:z:0*
T0*
_output_shapes
: r
)lambda/map/while/central_crop/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB 2       @°
'lambda/map/while/central_crop/truediv_1RealDiv'lambda/map/while/central_crop/sub_1:z:02lambda/map/while/central_crop/truediv_1/y:output:0*
T0*
_output_shapes
: 
$lambda/map/while/central_crop/Cast_5Cast+lambda/map/while/central_crop/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: g
%lambda/map/while/central_crop/mul_2/yConst*
_output_shapes
: *
dtype0*
value	B :¥
#lambda/map/while/central_crop/mul_2Mul(lambda/map/while/central_crop/Cast_2:y:0.lambda/map/while/central_crop/mul_2/y:output:0*
T0*
_output_shapes
: ¬
#lambda/map/while/central_crop/sub_2Sub6lambda/map/while/central_crop/strided_slice_1:output:0'lambda/map/while/central_crop/mul_2:z:0*
T0*
_output_shapes
: g
%lambda/map/while/central_crop/mul_3/yConst*
_output_shapes
: *
dtype0*
value	B :¥
#lambda/map/while/central_crop/mul_3Mul(lambda/map/while/central_crop/Cast_5:y:0.lambda/map/while/central_crop/mul_3/y:output:0*
T0*
_output_shapes
: ¬
#lambda/map/while/central_crop/sub_3Sub6lambda/map/while/central_crop/strided_slice_2:output:0'lambda/map/while/central_crop/mul_3:z:0*
T0*
_output_shapes
: g
%lambda/map/while/central_crop/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Ý
#lambda/map/while/central_crop/stackPack(lambda/map/while/central_crop/Cast_2:y:0(lambda/map/while/central_crop/Cast_5:y:0.lambda/map/while/central_crop/stack/2:output:0*
N*
T0*
_output_shapes
:r
'lambda/map/while/central_crop/stack_1/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿß
%lambda/map/while/central_crop/stack_1Pack'lambda/map/while/central_crop/sub_2:z:0'lambda/map/while/central_crop/sub_3:z:00lambda/map/while/central_crop/stack_1/2:output:0*
N*
T0*
_output_shapes
:
#lambda/map/while/central_crop/SliceSlice4lambda/map/while/adjust_saturation/Identity:output:0,lambda/map/while/central_crop/stack:output:0.lambda/map/while/central_crop/stack_1:output:0*
Index0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
&lambda/map/while/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ò
"lambda/map/while/resize/ExpandDims
ExpandDims,lambda/map/while/central_crop/Slice:output:0/lambda/map/while/resize/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿm
lambda/map/while/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   Ù
&lambda/map/while/resize/ResizeBilinearResizeBilinear+lambda/map/while/resize/ExpandDims:output:0%lambda/map/while/resize/size:output:0*
T0*(
_output_shapes
:àà*
half_pixel_centers(©
lambda/map/while/resize/SqueezeSqueeze7lambda/map/while/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:àà*
squeeze_dims
 ò
5lambda/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlambda_map_while_placeholder_1lambda_map_while_placeholder(lambda/map/while/resize/Squeeze:output:0*
_output_shapes
: *
element_dtype0:éèÒX
lambda/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :}
lambda/map/while/addAddV2lambda_map_while_placeholderlambda/map/while/add/y:output:0*
T0*
_output_shapes
: Z
lambda/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lambda/map/while/add_1AddV2.lambda_map_while_lambda_map_while_loop_counter!lambda/map/while/add_1/y:output:0*
T0*
_output_shapes
: z
lambda/map/while/IdentityIdentitylambda/map/while/add_1:z:0^lambda/map/while/NoOp*
T0*
_output_shapes
: 
lambda/map/while/Identity_1Identity)lambda_map_while_lambda_map_strided_slice^lambda/map/while/NoOp*
T0*
_output_shapes
: z
lambda/map/while/Identity_2Identitylambda/map/while/add:z:0^lambda/map/while/NoOp*
T0*
_output_shapes
: º
lambda/map/while/Identity_3IdentityElambda/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lambda/map/while/NoOp*
T0*
_output_shapes
: :éèÒ 
lambda/map/while/NoOpNoOpA^lambda/map/while/central_crop/assert_greater_equal/Assert/AssertH^lambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertK^lambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertR^lambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertH^lambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertO^lambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "?
lambda_map_while_identity"lambda/map/while/Identity:output:0"C
lambda_map_while_identity_1$lambda/map/while/Identity_1:output:0"C
lambda_map_while_identity_2$lambda/map/while/Identity_2:output:0"C
lambda_map_while_identity_3$lambda/map/while/Identity_3:output:0"\
+lambda_map_while_lambda_map_strided_slice_1-lambda_map_while_lambda_map_strided_slice_1_0"Ô
glambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensorilambda_map_while_tensorarrayv2read_tensorlistgetitem_lambda_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2
@lambda/map/while/central_crop/assert_greater_equal/Assert/Assert@lambda/map/while/central_crop/assert_greater_equal/Assert/Assert2
Glambda/map/while/central_crop/assert_positive/assert_less/Assert/AssertGlambda/map/while/central_crop/assert_positive/assert_less/Assert/Assert2
Jlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/AssertJlambda/map/while/random_flip_left_right/assert_greater_equal/Assert/Assert2¦
Qlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/AssertQlambda/map/while/random_flip_left_right/assert_positive/assert_less/Assert/Assert2
Glambda/map/while/random_flip_up_down/assert_greater_equal/Assert/AssertGlambda/map/while/random_flip_up_down/assert_greater_equal/Assert/Assert2 
Nlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/AssertNlambda/map/while/random_flip_up_down/assert_positive/assert_less/Assert/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¯
_
C__inference_CLASSES_layer_call_and_return_conditional_losses_158719

inputs
identity	R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :a
ArgMaxArgMaxinputsArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
IdentityIdentityArgMax:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨Z
¦
A__inference_vgg16_layer_call_and_return_conditional_losses_155352
input_1-
block1_conv1_155280:@!
block1_conv1_155282:@-
block1_conv2_155285:@@!
block1_conv2_155287:@.
block2_conv1_155291:@"
block2_conv1_155293:	/
block2_conv2_155296:"
block2_conv2_155298:	/
block3_conv1_155302:"
block3_conv1_155304:	/
block3_conv2_155307:"
block3_conv2_155309:	/
block3_conv3_155312:"
block3_conv3_155314:	/
block4_conv1_155318:"
block4_conv1_155320:	/
block4_conv2_155323:"
block4_conv2_155325:	/
block4_conv3_155328:"
block4_conv3_155330:	/
block5_conv1_155334:"
block5_conv1_155336:	/
block5_conv2_155339:"
block5_conv2_155341:	/
block5_conv3_155344:"
block5_conv3_155346:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_155280block1_conv1_155282*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_155285block1_conv2_155287*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_155291block2_conv1_155293*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_155296block2_conv2_155298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_155302block3_conv1_155304*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_155307block3_conv2_155309*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_155312block3_conv3_155314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_155318block4_conv1_155320*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_155323block4_conv2_155325*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_155328block4_conv3_155330*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_155334block5_conv1_155336*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_155339block5_conv2_155341*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_155344block5_conv3_155346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595ú
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
!
_user_specified_name	input_1
Ö
Ý
3lambda_map_while_random_flip_left_right_true_157207p
llambda_map_while_random_flip_left_right_reversev2_lambda_map_while_random_flip_left_right_control_dependency4
0lambda_map_while_random_flip_left_right_identity
6lambda/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:¬
1lambda/map/while/random_flip_left_right/ReverseV2	ReverseV2llambda_map_while_random_flip_left_right_reversev2_lambda_map_while_random_flip_left_right_control_dependency?lambda/map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
0lambda/map/while/random_flip_left_right/IdentityIdentity:lambda/map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"m
0lambda_map_while_random_flip_left_right_identity9lambda/map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ø

map_while_cond_156387$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_156387___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

D
(__inference_CLASSES_layer_call_fn_158708

inputs
identity	­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156267\
IdentityIdentityPartitionedCall:output:0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block1_conv2_layer_call_and_return_conditional_losses_159103

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs
³
H
,__inference_block4_pool_layer_call_fn_159298

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¥
-__inference_block5_conv1_layer_call_fn_159312

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥Z
¥
A__inference_vgg16_layer_call_and_return_conditional_losses_154846

inputs-
block1_conv1_154630:@!
block1_conv1_154632:@-
block1_conv2_154647:@@!
block1_conv2_154649:@.
block2_conv1_154665:@"
block2_conv1_154667:	/
block2_conv2_154682:"
block2_conv2_154684:	/
block3_conv1_154700:"
block3_conv1_154702:	/
block3_conv2_154717:"
block3_conv2_154719:	/
block3_conv3_154734:"
block3_conv3_154736:	/
block4_conv1_154752:"
block4_conv1_154754:	/
block4_conv2_154769:"
block4_conv2_154771:	/
block4_conv3_154786:"
block4_conv3_154788:	/
block5_conv1_154804:"
block5_conv1_154806:	/
block5_conv2_154821:"
block5_conv2_154823:	/
block5_conv3_154838:"
block5_conv3_154840:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_154630block1_conv1_154632*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_154647block1_conv2_154649*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_154665block2_conv1_154667*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_154682block2_conv2_154684*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_154700block3_conv1_154702*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_154717block3_conv2_154719*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_154734block3_conv3_154736*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_154752block4_conv1_154754*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_154769block4_conv2_154771*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_154786block4_conv3_154788*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_154804block5_conv1_154806*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_154821block5_conv2_154823*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_154838block5_conv3_154840*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595ú
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs


H__inference_block5_conv3_layer_call_and_return_conditional_losses_159363

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
õ
+__inference_sequential_layer_call_fn_155564
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155505o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
%
_user_specified_namevgg16_input
ú
Á
,map_while_random_flip_left_right_true_156026b
^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityy
/map/while/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
*map/while/random_flip_left_right/ReverseV2	ReverseV2^map_while_random_flip_left_right_reversev2_map_while_random_flip_left_right_control_dependency8map/while/random_flip_left_right/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
)map/while/random_flip_left_right/IdentityIdentity3map/while/random_flip_left_right/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³
ø
$__inference_signature_wrapper_157839	
bytes!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:
identity	

identity_1¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallbytesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2	*
_collective_manager_ids
 *6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_154538k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namebytes
ù
e
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156259

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block2_conv1_layer_call_and_return_conditional_losses_159133

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@
 
_user_specified_nameinputs
û
µ
*map_while_random_flip_up_down_false_156074[
Wmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityÊ
&map/while/random_flip_up_down/IdentityIdentityWmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ø

map_while_cond_158110$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice<
8map_while_map_while_cond_158110___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
µ
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_159384

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ó
A__inference_dense_layer_call_and_return_conditional_losses_159063

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


H__inference_block3_conv3_layer_call_and_return_conditional_losses_159223

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

c
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Á
-map_while_random_flip_left_right_false_156449a
]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency-
)map_while_random_flip_left_right_identityÓ
)map/while/random_flip_left_right/IdentityIdentity]map_while_random_flip_left_right_identity_map_while_random_flip_left_right_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"_
)map_while_random_flip_left_right_identity2map/while/random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ëÔ
ä
A__inference_model_layer_call_and_return_conditional_losses_157485

inputsV
<sequential_vgg16_block1_conv1_conv2d_readvariableop_resource:@K
=sequential_vgg16_block1_conv1_biasadd_readvariableop_resource:@V
<sequential_vgg16_block1_conv2_conv2d_readvariableop_resource:@@K
=sequential_vgg16_block1_conv2_biasadd_readvariableop_resource:@W
<sequential_vgg16_block2_conv1_conv2d_readvariableop_resource:@L
=sequential_vgg16_block2_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block2_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block2_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block3_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block3_conv3_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block4_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block4_conv3_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv1_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv1_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv2_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv2_biasadd_readvariableop_resource:	X
<sequential_vgg16_block5_conv3_conv2d_readvariableop_resource:L
=sequential_vgg16_block5_conv3_biasadd_readvariableop_resource:	B
/sequential_dense_matmul_readvariableop_resource:	>
0sequential_dense_biasadd_readvariableop_resource:
identity	

identity_1¢lambda/map/while¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp¢4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp¢3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpF
lambda/map/ShapeShapeinputs*
T0*
_output_shapes
:h
lambda/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lambda/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lambda/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lambda/map/strided_sliceStridedSlicelambda/map/Shape:output:0'lambda/map/strided_slice/stack:output:0)lambda/map/strided_slice/stack_1:output:0)lambda/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
&lambda/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
lambda/map/TensorArrayV2TensorListReserve/lambda/map/TensorArrayV2/element_shape:output:0!lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖf
#lambda/map/TensorArrayUnstack/ConstConst*
_output_shapes
: *
dtype0*
valueB Ò
2lambda/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputs,lambda/map/TensorArrayUnstack/Const:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÖR
lambda/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : s
(lambda/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
lambda/map/TensorArrayV2_1TensorListReserve1lambda/map/TensorArrayV2_1/element_shape:output:0!lambda/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
lambda/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : °
lambda/map/whileWhile&lambda/map/while/loop_counter:output:0!lambda/map/strided_slice:output:0lambda/map/Const:output:0#lambda/map/TensorArrayV2_1:handle:0!lambda/map/strided_slice:output:0Blambda/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *(
body R
lambda_map_while_body_157147*(
cond R
lambda_map_while_cond_157146*
output_shapes
: : : : : : 
;lambda/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"à   à      é
-lambda/map/TensorArrayV2Stack/TensorListStackTensorListStacklambda/map/while:output:3Dlambda/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
element_dtype0¸
3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
$sequential/vgg16/block1_conv1/Conv2DConv2D6lambda/map/TensorArrayV2Stack/TensorListStack:tensor:0;sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
®
4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%sequential/vgg16/block1_conv1/BiasAddBiasAdd-sequential/vgg16/block1_conv1/Conv2D:output:0<sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"sequential/vgg16/block1_conv1/ReluRelu.sequential/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¸
3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
$sequential/vgg16/block1_conv2/Conv2DConv2D0sequential/vgg16/block1_conv1/Relu:activations:0;sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides
®
4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ù
%sequential/vgg16/block1_conv2/BiasAddBiasAdd-sequential/vgg16/block1_conv2/Conv2D:output:0<sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"sequential/vgg16/block1_conv2/ReluRelu.sequential/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@Î
$sequential/vgg16/block1_pool/MaxPoolMaxPool0sequential/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides
¹
3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ý
$sequential/vgg16/block2_conv1/Conv2DConv2D-sequential/vgg16/block1_pool/MaxPool:output:0;sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¯
4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block2_conv1/BiasAddBiasAdd-sequential/vgg16/block2_conv1/Conv2D:output:0<sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"sequential/vgg16/block2_conv1/ReluRelu.sequential/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppº
3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block2_conv2/Conv2DConv2D0sequential/vgg16/block2_conv1/Relu:activations:0;sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¯
4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block2_conv2/BiasAddBiasAdd-sequential/vgg16/block2_conv2/Conv2D:output:0<sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"sequential/vgg16/block2_conv2/ReluRelu.sequential/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÏ
$sequential/vgg16/block2_pool/MaxPoolMaxPool0sequential/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block3_conv1/Conv2DConv2D-sequential/vgg16/block2_pool/MaxPool:output:0;sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv1/BiasAddBiasAdd-sequential/vgg16/block3_conv1/Conv2D:output:0<sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv1/ReluRelu.sequential/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88º
3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block3_conv2/Conv2DConv2D0sequential/vgg16/block3_conv1/Relu:activations:0;sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv2/BiasAddBiasAdd-sequential/vgg16/block3_conv2/Conv2D:output:0<sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv2/ReluRelu.sequential/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88º
3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block3_conv3/Conv2DConv2D0sequential/vgg16/block3_conv2/Relu:activations:0;sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¯
4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block3_conv3/BiasAddBiasAdd-sequential/vgg16/block3_conv3/Conv2D:output:0<sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"sequential/vgg16/block3_conv3/ReluRelu.sequential/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ï
$sequential/vgg16/block3_pool/MaxPoolMaxPool0sequential/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block4_conv1/Conv2DConv2D-sequential/vgg16/block3_pool/MaxPool:output:0;sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv1/BiasAddBiasAdd-sequential/vgg16/block4_conv1/Conv2D:output:0<sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv1/ReluRelu.sequential/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block4_conv2/Conv2DConv2D0sequential/vgg16/block4_conv1/Relu:activations:0;sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv2/BiasAddBiasAdd-sequential/vgg16/block4_conv2/Conv2D:output:0<sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv2/ReluRelu.sequential/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block4_conv3/Conv2DConv2D0sequential/vgg16/block4_conv2/Relu:activations:0;sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block4_conv3/BiasAddBiasAdd-sequential/vgg16/block4_conv3/Conv2D:output:0<sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block4_conv3/ReluRelu.sequential/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$sequential/vgg16/block4_pool/MaxPoolMaxPool0sequential/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
º
3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
$sequential/vgg16/block5_conv1/Conv2DConv2D-sequential/vgg16/block4_pool/MaxPool:output:0;sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv1/BiasAddBiasAdd-sequential/vgg16/block5_conv1/Conv2D:output:0<sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv1/ReluRelu.sequential/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block5_conv2/Conv2DConv2D0sequential/vgg16/block5_conv1/Relu:activations:0;sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv2/BiasAddBiasAdd-sequential/vgg16/block5_conv2/Conv2D:output:0<sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv2/ReluRelu.sequential/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<sequential_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
$sequential/vgg16/block5_conv3/Conv2DConv2D0sequential/vgg16/block5_conv2/Relu:activations:0;sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp=sequential_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%sequential/vgg16/block5_conv3/BiasAddBiasAdd-sequential/vgg16/block5_conv3/Conv2D:output:0<sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/vgg16/block5_conv3/ReluRelu.sequential/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
$sequential/vgg16/block5_pool/MaxPoolMaxPool0sequential/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

@sequential/vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ã
.sequential/vgg16/global_average_pooling2d/MeanMean-sequential/vgg16/block5_pool/MaxPool:output:0Isequential/vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¼
sequential/dense/MatMulMatMul7sequential/vgg16/global_average_pooling2d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
CLASSES/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
CLASSES/ArgMaxArgMax"sequential/dense/Softmax:softmax:0!CLASSES/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityCLASSES/ArgMax:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_1Identity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp^lambda/map/while(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp5^sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp5^sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp4^sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
lambda/map/whilelambda/map/while2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2l
4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block1_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block1_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block1_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block1_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block2_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block2_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block2_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block2_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block3_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block3_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block4_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block4_conv3/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv1/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv1/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv2/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv2/Conv2D/ReadVariableOp2l
4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp4sequential/vgg16/block5_conv3/BiasAdd/ReadVariableOp2j
3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp3sequential/vgg16/block5_conv3/Conv2D/ReadVariableOp:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¢
-__inference_block1_conv2_layer_call_fn_159092

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
 
_user_specified_nameinputs
è
ö
A__inference_vgg16_layer_call_and_return_conditional_losses_159043

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpps
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88­
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      °
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

û
F__inference_sequential_layer_call_and_return_conditional_losses_155881
vgg16_input&
vgg16_155822:@
vgg16_155824:@&
vgg16_155826:@@
vgg16_155828:@'
vgg16_155830:@
vgg16_155832:	(
vgg16_155834:
vgg16_155836:	(
vgg16_155838:
vgg16_155840:	(
vgg16_155842:
vgg16_155844:	(
vgg16_155846:
vgg16_155848:	(
vgg16_155850:
vgg16_155852:	(
vgg16_155854:
vgg16_155856:	(
vgg16_155858:
vgg16_155860:	(
vgg16_155862:
vgg16_155864:	(
vgg16_155866:
vgg16_155868:	(
vgg16_155870:
vgg16_155872:	
dense_155875:	
dense_155877:
identity¢dense/StatefulPartitionedCall¢vgg16/StatefulPartitionedCallí
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_inputvgg16_155822vgg16_155824vgg16_155826vgg16_155828vgg16_155830vgg16_155832vgg16_155834vgg16_155836vgg16_155838vgg16_155840vgg16_155842vgg16_155844vgg16_155846vgg16_155848vgg16_155850vgg16_155852vgg16_155854vgg16_155856vgg16_155858vgg16_155860vgg16_155862vgg16_155864vgg16_155866vgg16_155868vgg16_155870vgg16_155872*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_vgg16_layer_call_and_return_conditional_losses_154846
dense/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0dense_155875dense_155877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_155498u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
%
_user_specified_namevgg16_input
³
H
,__inference_block2_pool_layer_call_fn_159158

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 


A__inference_model_layer_call_and_return_conditional_losses_156751

inputs+
sequential_156690:@
sequential_156692:@+
sequential_156694:@@
sequential_156696:@,
sequential_156698:@ 
sequential_156700:	-
sequential_156702: 
sequential_156704:	-
sequential_156706: 
sequential_156708:	-
sequential_156710: 
sequential_156712:	-
sequential_156714: 
sequential_156716:	-
sequential_156718: 
sequential_156720:	-
sequential_156722: 
sequential_156724:	-
sequential_156726: 
sequential_156728:	-
sequential_156730: 
sequential_156732:	-
sequential_156734: 
sequential_156736:	-
sequential_156738: 
sequential_156740:	$
sequential_156742:	
sequential_156744:
identity	

identity_1¢lambda/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallÑ
lambda/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_156618¾
"sequential/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0sequential_156690sequential_156692sequential_156694sequential_156696sequential_156698sequential_156700sequential_156702sequential_156704sequential_156706sequential_156708sequential_156710sequential_156712sequential_156714sequential_156716sequential_156718sequential_156720sequential_156722sequential_156724sequential_156726sequential_156728sequential_156730sequential_156732sequential_156734sequential_156736sequential_156738sequential_156740sequential_156742sequential_156744*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_155699ê
PROBABILITIES/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_156361Õ
CLASSES/PartitionedCallPartitionedCall&PROBABILITIES/PartitionedCall:output:0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_CLASSES_layer_call_and_return_conditional_losses_156346k
IdentityIdentity CLASSES/PartitionedCall:output:0^NoOp*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw

Identity_1Identity&PROBABILITIES/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lambda/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:K G
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
µ
*map_while_random_flip_up_down_false_158219[
Wmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityÊ
&map/while/random_flip_up_down/IdentityIdentityWmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥Z
¥
A__inference_vgg16_layer_call_and_return_conditional_losses_155165

inputs-
block1_conv1_155093:@!
block1_conv1_155095:@-
block1_conv2_155098:@@!
block1_conv2_155100:@.
block2_conv1_155104:@"
block2_conv1_155106:	/
block2_conv2_155109:"
block2_conv2_155111:	/
block3_conv1_155115:"
block3_conv1_155117:	/
block3_conv2_155120:"
block3_conv2_155122:	/
block3_conv3_155125:"
block3_conv3_155127:	/
block4_conv1_155131:"
block4_conv1_155133:	/
block4_conv2_155136:"
block4_conv2_155138:	/
block4_conv3_155141:"
block4_conv3_155143:	/
block5_conv1_155147:"
block5_conv1_155149:	/
block5_conv2_155152:"
block5_conv2_155154:	/
block5_conv3_155157:"
block5_conv3_155159:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_155093block1_conv1_155095*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv1_layer_call_and_return_conditional_losses_154629´
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_155098block1_conv2_155100*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_conv2_layer_call_and_return_conditional_losses_154646ð
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block1_pool_layer_call_and_return_conditional_losses_154547ª
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_155104block2_conv1_155106*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_154664³
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_155109block2_conv2_155111*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_154681ñ
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_154559ª
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_155115block3_conv1_155117*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv1_layer_call_and_return_conditional_losses_154699³
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_155120block3_conv2_155122*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv2_layer_call_and_return_conditional_losses_154716³
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_155125block3_conv3_155127*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_conv3_layer_call_and_return_conditional_losses_154733ñ
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block3_pool_layer_call_and_return_conditional_losses_154571ª
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_155131block4_conv1_155133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv1_layer_call_and_return_conditional_losses_154751³
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_155136block4_conv2_155138*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv2_layer_call_and_return_conditional_losses_154768³
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_155141block4_conv3_155143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_conv3_layer_call_and_return_conditional_losses_154785ñ
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block4_pool_layer_call_and_return_conditional_losses_154583ª
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_155147block5_conv1_155149*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv1_layer_call_and_return_conditional_losses_154803³
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_155152block5_conv2_155154*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv2_layer_call_and_return_conditional_losses_154820³
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_155157block5_conv3_155159*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_conv3_layer_call_and_return_conditional_losses_154837ñ
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_block5_pool_layer_call_and_return_conditional_losses_154595ú
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_154608
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
K
¦
__inference__traced_save_159522
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ø
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*¡
valueB'B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_variables/26/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_variables/27/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH»
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ø
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*£
_input_shapes
: :@:@:@@:@:@::::::::::::::::::::::	:: : : : : : : : :	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%%!

_output_shapes
:	: &

_output_shapes
::'

_output_shapes
: 
û
µ
*map_while_random_flip_up_down_false_156496[
Wmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityÊ
&map/while/random_flip_up_down/IdentityIdentityWmap_while_random_flip_up_down_identity_map_while_random_flip_up_down_control_dependency*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó
µ
)map_while_random_flip_up_down_true_158218\
Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency*
&map_while_random_flip_up_down_identityv
,map/while/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 
'map/while/random_flip_up_down/ReverseV2	ReverseV2Xmap_while_random_flip_up_down_reversev2_map_while_random_flip_up_down_control_dependency5map/while/random_flip_up_down/ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
&map/while/random_flip_up_down/IdentityIdentity0map/while/random_flip_up_down/ReverseV2:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"Y
&map_while_random_flip_up_down_identity/map/while/random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:: 6
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


H__inference_block3_conv2_layer_call_and_return_conditional_losses_159203

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*á
serving_defaultÍ
3
bytes*
serving_default_bytes:0ÿÿÿÿÿÿÿÿÿ7
CLASSES,
StatefulPartitionedCall:0	ÿÿÿÿÿÿÿÿÿA
PROBABILITIES0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ó
¯
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_model_layer_call_fn_156332
&__inference_model_layer_call_fn_157068
&__inference_model_layer_call_fn_157131
&__inference_model_layer_call_fn_156875À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_157485
A__inference_model_layer_call_and_return_conditional_losses_157774
A__inference_model_layer_call_and_return_conditional_losses_156940
A__inference_model_layer_call_and_return_conditional_losses_157005À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
!__inference__wrapped_model_154538bytes"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Jserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
'__inference_lambda_layer_call_fn_157844
'__inference_lambda_layer_call_fn_157849À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
B__inference_lambda_layer_call_and_return_conditional_losses_158095
B__inference_lambda_layer_call_and_return_conditional_losses_158341À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 

Player-0
Qlayer_with_weights-0
Qlayer-1
Rlayer_with_weights-1
Rlayer-2
Slayer-3
Tlayer_with_weights-2
Tlayer-4
Ulayer_with_weights-3
Ulayer-5
Vlayer-6
Wlayer_with_weights-4
Wlayer-7
Xlayer_with_weights-5
Xlayer-8
Ylayer_with_weights-6
Ylayer-9
Zlayer-10
[layer_with_weights-7
[layer-11
\layer_with_weights-8
\layer-12
]layer_with_weights-9
]layer-13
^layer-14
_layer_with_weights-10
_layer-15
`layer_with_weights-11
`layer-16
alayer_with_weights-12
alayer-17
blayer-18
clayer-19
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_network
»

Ckernel
Dbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
k
	pdecay
qlearning_rate
rmomentum
siterCmomentuméDmomentumê"
	optimizer
ö
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25
C26
D27"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ú2÷
+__inference_sequential_layer_call_fn_155564
+__inference_sequential_layer_call_fn_158406
+__inference_sequential_layer_call_fn_158467
+__inference_sequential_layer_call_fn_155819À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_158576
F__inference_sequential_layer_call_and_return_conditional_losses_158685
F__inference_sequential_layer_call_and_return_conditional_losses_155881
F__inference_sequential_layer_call_and_return_conditional_losses_155943À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
¦2£
.__inference_PROBABILITIES_layer_call_fn_158690
.__inference_PROBABILITIES_layer_call_fn_158695À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158699
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158703À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
2
(__inference_CLASSES_layer_call_fn_158708
(__inference_CLASSES_layer_call_fn_158713À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
C__inference_CLASSES_layer_call_and_return_conditional_losses_158719
C__inference_CLASSES_layer_call_and_return_conditional_losses_158725À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
/:-2block3_conv1/kernel
 :2block3_conv1/bias
/:-2block3_conv2/kernel
 :2block3_conv2/bias
/:-2block3_conv3/kernel
 :2block3_conv3/bias
/:-2block4_conv1/kernel
 :2block4_conv1/bias
/:-2block4_conv2/kernel
 :2block4_conv2/bias
/:-2block4_conv3/kernel
 :2block4_conv3/bias
/:-2block5_conv1/kernel
 :2block5_conv1/bias
/:-2block5_conv2/kernel
 :2block5_conv2/bias
/:-2block5_conv3/kernel
 :2block5_conv3/bias
:	2dense/kernel
:2
dense/bias
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÉBÆ
$__inference_signature_wrapper_157839bytes"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"
_tf_keras_input_layer
Á

)kernel
*bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

+kernel
,bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

-kernel
.bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

/kernel
0bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

1kernel
2bias
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

3kernel
4bias
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

5kernel
6bias
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"
_tf_keras_layer
«
¹	variables
ºtrainable_variables
»regularization_losses
¼	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

7kernel
8bias
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

9kernel
:bias
Å	variables
Ætrainable_variables
Çregularization_losses
È	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

;kernel
<bias
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

=kernel
>bias
×	variables
Øtrainable_variables
Ùregularization_losses
Ú	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

?kernel
@bias
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Akernel
Bbias
ã	variables
ätrainable_variables
åregularization_losses
æ	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"
_tf_keras_layer
«
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_vgg16_layer_call_fn_154901
&__inference_vgg16_layer_call_fn_158782
&__inference_vgg16_layer_call_fn_158839
&__inference_vgg16_layer_call_fn_155277À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_vgg16_layer_call_and_return_conditional_losses_158941
A__inference_vgg16_layer_call_and_return_conditional_losses_159043
A__inference_vgg16_layer_call_and_return_conditional_losses_155352
A__inference_vgg16_layer_call_and_return_conditional_losses_155427À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_layer_call_fn_159052¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_159063¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
ÿ0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block1_conv1_layer_call_fn_159072¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block1_conv1_layer_call_and_return_conditional_losses_159083¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block1_conv2_layer_call_fn_159092¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block1_conv2_layer_call_and_return_conditional_losses_159103¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block1_pool_layer_call_fn_159108¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block1_pool_layer_call_and_return_conditional_losses_159113¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block2_conv1_layer_call_fn_159122¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block2_conv1_layer_call_and_return_conditional_losses_159133¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block2_conv2_layer_call_fn_159142¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block2_conv2_layer_call_and_return_conditional_losses_159153¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block2_pool_layer_call_fn_159158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block2_pool_layer_call_and_return_conditional_losses_159163¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv1_layer_call_fn_159172¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv1_layer_call_and_return_conditional_losses_159183¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv2_layer_call_fn_159192¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv2_layer_call_and_return_conditional_losses_159203¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_conv3_layer_call_fn_159212¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_conv3_layer_call_and_return_conditional_losses_159223¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
¹	variables
ºtrainable_variables
»regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block3_pool_layer_call_fn_159228¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block3_pool_layer_call_and_return_conditional_losses_159233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv1_layer_call_fn_159242¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv1_layer_call_and_return_conditional_losses_159253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
Å	variables
Ætrainable_variables
Çregularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv2_layer_call_fn_159262¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv2_layer_call_and_return_conditional_losses_159273¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_conv3_layer_call_fn_159282¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_conv3_layer_call_and_return_conditional_losses_159293¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block4_pool_layer_call_fn_159298¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block4_pool_layer_call_and_return_conditional_losses_159303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
×	variables
Øtrainable_variables
Ùregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv1_layer_call_fn_159312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv1_layer_call_and_return_conditional_losses_159323¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv2_layer_call_fn_159332¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv2_layer_call_and_return_conditional_losses_159343¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
ã	variables
ätrainable_variables
åregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_conv3_layer_call_fn_159352¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_conv3_layer_call_and_return_conditional_losses_159363¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_block5_pool_layer_call_fn_159368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_block5_pool_layer_call_and_return_conditional_losses_159373¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
ã2à
9__inference_global_average_pooling2d_layer_call_fn_159378¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_159384¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
æ
)0
*1
+2
,3
-4
.5
/6
07
18
29
310
411
512
613
714
815
916
:17
;18
<19
=20
>21
?22
@23
A24
B25"
trackable_list_wrapper
¶
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
`16
a17
b18
c19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

àtotal

ácount
â	variables
ã	keras_api"
_tf_keras_metric
c

ätotal

åcount
æ
_fn_kwargs
ç	variables
è	keras_api"
_tf_keras_metric
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
à0
á1"
trackable_list_wrapper
.
â	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ä0
å1"
trackable_list_wrapper
.
ç	variables"
_generic_user_object
*:(	2SGD/dense/kernel/momentum
#:!2SGD/dense/bias/momentum£
C__inference_CLASSES_layer_call_and_return_conditional_losses_158719\7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "!¢

0ÿÿÿÿÿÿÿÿÿ	
 £
C__inference_CLASSES_layer_call_and_return_conditional_losses_158725\7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "!¢

0ÿÿÿÿÿÿÿÿÿ	
 {
(__inference_CLASSES_layer_call_fn_158708O7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "ÿÿÿÿÿÿÿÿÿ	{
(__inference_CLASSES_layer_call_fn_158713O7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "ÿÿÿÿÿÿÿÿÿ	­
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158699`7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
I__inference_PROBABILITIES_layer_call_and_return_conditional_losses_158703`7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_PROBABILITIES_layer_call_fn_158690S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_PROBABILITIES_layer_call_fn_158695S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "ÿÿÿÿÿÿÿÿÿÙ
!__inference__wrapped_model_154538³)*+,-./0123456789:;<=>?@ABCD*¢'
 ¢

bytesÿÿÿÿÿÿÿÿÿ
ª "gªd
(
CLASSES
CLASSESÿÿÿÿÿÿÿÿÿ	
8
PROBABILITIES'$
PROBABILITIESÿÿÿÿÿÿÿÿÿ¼
H__inference_block1_conv1_layer_call_and_return_conditional_losses_159083p)*9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà@
 
-__inference_block1_conv1_layer_call_fn_159072c)*9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª ""ÿÿÿÿÿÿÿÿÿàà@¼
H__inference_block1_conv2_layer_call_and_return_conditional_losses_159103p+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà@
 
-__inference_block1_conv2_layer_call_fn_159092c+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà@
ª ""ÿÿÿÿÿÿÿÿÿàà@ê
G__inference_block1_pool_layer_call_and_return_conditional_losses_159113R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block1_pool_layer_call_fn_159108R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
H__inference_block2_conv1_layer_call_and_return_conditional_losses_159133m-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
-__inference_block2_conv1_layer_call_fn_159122`-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp@
ª "!ÿÿÿÿÿÿÿÿÿppº
H__inference_block2_conv2_layer_call_and_return_conditional_losses_159153n/08¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿpp
 
-__inference_block2_conv2_layer_call_fn_159142a/08¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿpp
ª "!ÿÿÿÿÿÿÿÿÿppê
G__inference_block2_pool_layer_call_and_return_conditional_losses_159163R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block2_pool_layer_call_fn_159158R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block3_conv1_layer_call_and_return_conditional_losses_159183n128¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
-__inference_block3_conv1_layer_call_fn_159172a128¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88º
H__inference_block3_conv2_layer_call_and_return_conditional_losses_159203n348¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
-__inference_block3_conv2_layer_call_fn_159192a348¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88º
H__inference_block3_conv3_layer_call_and_return_conditional_losses_159223n568¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ88
 
-__inference_block3_conv3_layer_call_fn_159212a568¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ88
ª "!ÿÿÿÿÿÿÿÿÿ88ê
G__inference_block3_pool_layer_call_and_return_conditional_losses_159233R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block3_pool_layer_call_fn_159228R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv1_layer_call_and_return_conditional_losses_159253n788¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv1_layer_call_fn_159242a788¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv2_layer_call_and_return_conditional_losses_159273n9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv2_layer_call_fn_159262a9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block4_conv3_layer_call_and_return_conditional_losses_159293n;<8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block4_conv3_layer_call_fn_159282a;<8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿê
G__inference_block4_pool_layer_call_and_return_conditional_losses_159303R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block4_pool_layer_call_fn_159298R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv1_layer_call_and_return_conditional_losses_159323n=>8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv1_layer_call_fn_159312a=>8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv2_layer_call_and_return_conditional_losses_159343n?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv2_layer_call_fn_159332a?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿº
H__inference_block5_conv3_layer_call_and_return_conditional_losses_159363nAB8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_block5_conv3_layer_call_fn_159352aAB8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿê
G__inference_block5_pool_layer_call_and_return_conditional_losses_159373R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_block5_pool_layer_call_fn_159368R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
A__inference_dense_layer_call_and_return_conditional_losses_159063]CD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_dense_layer_call_fn_159052PCD0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÝ
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_159384R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ´
9__inference_global_average_pooling2d_layer_call_fn_159378wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
B__inference_lambda_layer_call_and_return_conditional_losses_158095f3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 ¬
B__inference_lambda_layer_call_and_return_conditional_losses_158341f3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 
'__inference_lambda_layer_call_fn_157844Y3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª ""ÿÿÿÿÿÿÿÿÿàà
'__inference_lambda_layer_call_fn_157849Y3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ

 
p
ª ""ÿÿÿÿÿÿÿÿÿààá
A__inference_model_layer_call_and_return_conditional_losses_156940)*+,-./0123456789:;<=>?@ABCD2¢/
(¢%

bytesÿÿÿÿÿÿÿÿÿ
p 

 
ª "G¢D
=:

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 á
A__inference_model_layer_call_and_return_conditional_losses_157005)*+,-./0123456789:;<=>?@ABCD2¢/
(¢%

bytesÿÿÿÿÿÿÿÿÿ
p

 
ª "G¢D
=:

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 â
A__inference_model_layer_call_and_return_conditional_losses_157485)*+,-./0123456789:;<=>?@ABCD3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "G¢D
=:

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 â
A__inference_model_layer_call_and_return_conditional_losses_157774)*+,-./0123456789:;<=>?@ABCD3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "G¢D
=:

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 ¸
&__inference_model_layer_call_fn_156332)*+,-./0123456789:;<=>?@ABCD2¢/
(¢%

bytesÿÿÿÿÿÿÿÿÿ
p 

 
ª "96

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿ¸
&__inference_model_layer_call_fn_156875)*+,-./0123456789:;<=>?@ABCD2¢/
(¢%

bytesÿÿÿÿÿÿÿÿÿ
p

 
ª "96

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿ¹
&__inference_model_layer_call_fn_157068)*+,-./0123456789:;<=>?@ABCD3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "96

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿ¹
&__inference_model_layer_call_fn_157131)*+,-./0123456789:;<=>?@ABCD3¢0
)¢&

inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "96

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿØ
F__inference_sequential_layer_call_and_return_conditional_losses_155881)*+,-./0123456789:;<=>?@ABCDF¢C
<¢9
/,
vgg16_inputÿÿÿÿÿÿÿÿÿàà
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
F__inference_sequential_layer_call_and_return_conditional_losses_155943)*+,-./0123456789:;<=>?@ABCDF¢C
<¢9
/,
vgg16_inputÿÿÿÿÿÿÿÿÿàà
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ó
F__inference_sequential_layer_call_and_return_conditional_losses_158576)*+,-./0123456789:;<=>?@ABCDA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ó
F__inference_sequential_layer_call_and_return_conditional_losses_158685)*+,-./0123456789:;<=>?@ABCDA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
+__inference_sequential_layer_call_fn_155564)*+,-./0123456789:;<=>?@ABCDF¢C
<¢9
/,
vgg16_inputÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
+__inference_sequential_layer_call_fn_155819)*+,-./0123456789:;<=>?@ABCDF¢C
<¢9
/,
vgg16_inputÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿª
+__inference_sequential_layer_call_fn_158406{)*+,-./0123456789:;<=>?@ABCDA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿª
+__inference_sequential_layer_call_fn_158467{)*+,-./0123456789:;<=>?@ABCDA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿå
$__inference_signature_wrapper_157839¼)*+,-./0123456789:;<=>?@ABCD3¢0
¢ 
)ª&
$
bytes
bytesÿÿÿÿÿÿÿÿÿ"gªd
(
CLASSES
CLASSESÿÿÿÿÿÿÿÿÿ	
8
PROBABILITIES'$
PROBABILITIESÿÿÿÿÿÿÿÿÿÎ
A__inference_vgg16_layer_call_and_return_conditional_losses_155352)*+,-./0123456789:;<=>?@ABB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Î
A__inference_vgg16_layer_call_and_return_conditional_losses_155427)*+,-./0123456789:;<=>?@ABB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Í
A__inference_vgg16_layer_call_and_return_conditional_losses_158941)*+,-./0123456789:;<=>?@ABA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Í
A__inference_vgg16_layer_call_and_return_conditional_losses_159043)*+,-./0123456789:;<=>?@ABA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
&__inference_vgg16_layer_call_fn_154901{)*+,-./0123456789:;<=>?@ABB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¥
&__inference_vgg16_layer_call_fn_155277{)*+,-./0123456789:;<=>?@ABB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿ¤
&__inference_vgg16_layer_call_fn_158782z)*+,-./0123456789:;<=>?@ABA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
&__inference_vgg16_layer_call_fn_158839z)*+,-./0123456789:;<=>?@ABA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "ÿÿÿÿÿÿÿÿÿ
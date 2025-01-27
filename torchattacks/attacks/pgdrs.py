import torch
import torch.nn.functional as F
import copy
import tensorflow as tf
import numpy as np
from ..attack import Attack


class Noise():
    def __init__(self, noise_type, noise_sd):
        self.noise_type = noise_type
        self.noise_sd = noise_sd

    def __call__(self, img):
        if self.noise_type == "guassian":
            noise = torch.randn_like(img.float())*self.noise_sd
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(img.float()) - 0.5)*2*self.noise_sd
        return noise


class PGDRS(Attack):
    r"""
    PGD for randmized smoothing in the paper 'Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers'
    [https://arxiv.org/abs/1906.04584]
    Modification of the code from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        noise_type (str): guassian or uniform. (Default: guassian)
        noise_sd (float): standard deviation for normal distributio, or range for . (Default: 0.5)
        noise_batch_size (int): guassian or uniform. (Default: 5)
        batch_max (int): split data into small chunk if the total number of augmented data points, len(inputs)*noise_batch_size, are larger than batch_max, in case GPU memory is insufficient. (Default: 2048)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDRS(model, eps=8/255, alpha=2/255, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048):
        super().__init__('PGDRS', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.noise_func = Noise(noise_type, noise_sd)
        self.noise_batch_size = noise_batch_size
        self.supported_mode = ['default', 'targeted']
        self.batch_max = batch_max

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0]*self.noise_batch_size > self.batch_max:
            split_num = int(self.batch_max/self.noise_batch_size)
            inputs_split = torch.split(inputs,
                                       split_size_or_sections=split_num)
            labels_split = torch.split(labels,
                                       split_size_or_sections=split_num)
            img_list = []
            for img_sub, lab_sub in zip(inputs_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(inputs, labels)

    def _forward(self, images, labels):
        r"""
        Overridden.
        """

        image = images.clone().detach().to(self.device)
        label = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_label = self.get_target_label(image, label)
            
            
        # expend the inputs over noise_batch_size
        shape1 = torch.Size([image.shape[0], self.noise_batch_size]) + image.shape[1:]  # nopep8
        inputs_exp11 = image.unsqueeze(1).expand(shape1)
        inputs_exp1 = inputs_exp11.reshape(torch.Size([-1]) + inputs_exp11.shape[2:])  # nopep8

        delta1 = torch.zeros(
            (len(label), *inputs_exp1.shape[1:]), requires_grad=True, device=self.device)
        delta_last1 = torch.zeros(
            (len(label), *inputs_exp1.shape[1:]), requires_grad=False, device=self.device)

        for _ in range(self.steps):
            delta1.requires_grad = True
            # img_adv is the perturbed data for randmized smoothing
            # delta.repeat(1,self.noise_batch_size,1,1).view_as(inputs_exp)
            img_adv1 = inputs_exp1 + delta1.unsqueeze(1).repeat((1, self.noise_batch_size, 1, 1, 1)).view_as(inputs_exp1)  # nopep8
            img_adv1 = torch.clamp(img_adv1, min=0, max=1)
 
            noise_added1 = self.noise_func(img_adv1.view(len(img_adv1), -1))
            noise_added1 = noise_added1.view(img_adv1.shape)

            adv_noise1 = img_adv1 + noise_added1
        #     logits = self.get_logits(adv_noise)

        #     softmax = F.softmax(logits, dim=1)
        #     # average the probabilities across noise
        #     average_softmax = softmax.reshape(
        #         -1, self.noise_batch_size, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
        #     logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
        #     ce_loss = F.nll_loss(
        #         logsoftmax, labels) if not self.targeted else -F.nll_loss(logsoftmax, target_labels)

        #     grad = torch.autograd.grad(
        #         ce_loss, delta1, retain_graph=False, create_graph=False)[0]
        #     delta = delta_last + self.alpha*torch.sign(grad)
        #     delta = torch.clamp(delta1, min=-self.eps, max=self.eps)
        #     delta_last.data = copy.deepcopy(delta.data)

        # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        
        ###############################################
        ##  tf code #################
        ######################################
        images_torch = images.clone().detach().to(self.device)
        labels_torch = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images_torch, labels_torch)    
            target_labels = tf.convert_to_tensor(target_labels.numpy()).to(self.device)
        
        images = tf.convert_to_tensor(images.numpy())
        images = tf.cast(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels.numpy())
        labels = tf.cast(labels, dtype=tf.float32)

        # Expand the inputs over noise_batch_size
        shape = tf.concat([[tf.shape(images)[0], self.noise_batch_size], tf.shape(images)[1:]], axis=0)
        inputs_exp = tf.expand_dims(images, axis=1)
        inputs_exp = tf.tile(inputs_exp, [1, self.noise_batch_size, 1, 1, 1])
        # Get the dynamic shape of the inputs_exp tensor
        dynamic_shape = tf.concat([[-1], shape[2:]], axis=0)
        inputs_exp = tf.reshape(inputs_exp, dynamic_shape)
         
        delta = tf.Variable(tf.zeros((len(labels), *inputs_exp.shape[1:])), dtype=tf.float32, trainable=True)
        delta_last = tf.Variable(tf.zeros((len(labels), *inputs_exp.shape[1:])), dtype=tf.float32, trainable=False)
        # Initialize 'delta' with small random values
        # delta = tf.random.normal(
        #     shape=(tf.shape(labels)[0], *tf.shape(inputs_exp)[1:]), mean=0.0, stddev=0.01, dtype=tf.float32)
        
                
        for _ in range(self.steps):  
            # Create `img_adv` with the same shape as in the original PyTorch code
            # img_adv = inputs_exp + tf.repeat(tf.expand_dims(delta, axis=1), self.noise_batch_size, axis=1)
            # Make it a trainable variable by wrapping it with `tf.Variable`
            
            expanded_delta = delta[:, tf.newaxis]
            expanded_delta_repeated = tf.repeat(expanded_delta, self.noise_batch_size, axis=1)
            expanded_delta_repeated = tf.reshape(expanded_delta_repeated, inputs_exp.shape)
            img_adv = inputs_exp + expanded_delta_repeated
            img_adv = tf.clip_by_value(img_adv, clip_value_min=0, clip_value_max=1)
            

            
            # Convert img_adv to a torch tensor and move it to the GPU device
            img_adv_torch = img_adv.numpy()
            img_adv_torch = torch.from_numpy(img_adv_torch).to(self.device)
            # Do calculation
            noise_added  = self.noise_func(img_adv_torch.view(len(img_adv_torch), -1))
            noise_added = noise_added.view(img_adv_torch.shape)
            # Convert adv_noise to a tf tensor
            noise_added  = tf.constant(noise_added.cpu().numpy(), dtype=tf.float32) 
            
            adv_noise = img_adv + noise_added
             
   
            loss = tf.keras.losses.categorical_crossentropy
            # loss = tf.keras.losses.CategoricalCrossentropy()


                                
            with tf.device('/gpu:0'):   
                with tf.GradientTape(persistent=True) as tape:
                    delta = tf.Variable(delta, dtype=tf.float32, trainable=True)
                    tape.watch(adv_noise)
                    tape.watch(delta)
                    
                    logits = self.get_logits(adv_noise)
                     
                    softmax = tf.nn.softmax(logits, axis=1)
                    # average the probabilities across noise
                    average_softmax = tf.reshape(softmax, (-1, self.noise_batch_size, tf.shape(logits)[-1]))
                    average_softmax = tf.reduce_mean(average_softmax, axis=1, keepdims=True)
                    average_softmax = tf.squeeze(average_softmax, axis=1)
                    logsoftmax = tf.math.log(tf.clip_by_value(average_softmax, clip_value_min=1e-20, clip_value_max=np.inf)) 
             
                    # Calculate the loss
                    if self.targeted:
                        cost = -loss(target_labels, logsoftmax)
                    else:
                        cost = loss(labels, logsoftmax)
                       
            grad = tape.gradient(cost, delta)
            delta = delta_last + self.alpha * tf.sign(grad)
            delta = tf.clip_by_value(delta, clip_value_min=-self.eps, clip_value_max=self.eps)
            delta_last.assign(delta)
            
        adv_images = tf.clip_by_value(images + delta, clip_value_min=0, clip_value_max=1)
        adv_images = tf.stop_gradient(adv_images)
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device)
        
        return adv_images_torch

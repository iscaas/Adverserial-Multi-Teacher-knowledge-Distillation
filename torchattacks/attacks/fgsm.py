import torch
import torch.nn as nn
import tensorflow as tf
from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, device=None, eps=8/255, alpha=2/225, steps=2):
        super().__init__('FGSM', model, device)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        # images.requires_grad = True
        # outputs = self.get_logits(images)

        # # Calculate loss
        # if self.targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        #     cost = loss(outputs, labels)

        # # Update adversarial images
        # grad = torch.autograd.grad(cost, images,
        #                            retain_graph=False, create_graph=False)[0]

        # adv_images = images + self.eps*grad.sign()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        ######################################################################
        ##  tf code #################
       ############################################################################
        images_torch = images.clone().detach().to(self.device)
        labels_torch = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images_torch, labels_torch)    
            target_labels = tf.convert_to_tensor(target_labels.numpy()).to(self.device)

        images = tf.convert_to_tensor(images.numpy())
        labels = tf.convert_to_tensor(labels.numpy())
        loss = tf.keras.losses.CategoricalCrossentropy()
        
        images = tf.cast(images, dtype=tf.float32)
        images = tf.Variable(images, trainable=True)
        with tf.device('/gpu:0'):   
            with tf.GradientTape() as tape:
                tape.watch(images)
                outputs = self.get_logits(images)
                # Calculate the loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
        
        grad = tape.gradient(cost, images)
        adv_images = images +  self.eps * tf.sign(grad)
        adv_images = tf.clip_by_value(adv_images, clip_value_min=0, clip_value_max=1)
        adv_images = tf.stop_gradient(adv_images)
        
        
        # Convert adv_images to a torch tensor and move it to the GPU device
        adv_images_np = adv_images.numpy()
        adv_images_torch = torch.from_numpy(adv_images_np).to(self.device)
        
        return adv_images_torch

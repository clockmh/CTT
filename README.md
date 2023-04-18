# CTT - Correlation-based Transformer Tracking [ICANN2022]
Official implementation of the CTT (ICANN2022891267) , only training code.


## Tracker
#### CTT ####

[**[Paper]**](https://link.springer.com/chapter/10.1007/978-3-031-15919-0_8)

In recent studies on object tracking, Siamese tracking has achieved state-of-the-art performance due to its robustness and accuracy. Cross-correlation which is responsible for calculating similarity plays an important role in the development of Siamese tracking. However, the fact that general cross-correlation is a local operation leads to the lack of global contextual information. Although introducing transformer into tracking seems helpful to gain more semantic information, it will also bring more background interference, thus leads to the decline of the accuracy especially in long-term tracking. To address these problems, we propose a novel tracker, which adopts transformer architecture combined with cross-correlation, referred as correlation-based transformer tracking (CTT). When capturing global contextual information, the proposed CTT takes advantage of cross-correlation for more accurate feature fusion. This architecture is helpful to improve the tracking performance, especially long-term tracking. Extensive experimental results on large-scale benchmark datasets show that the proposed CTT achieves state-of-the-art performance, and particularly performs better than other trackers in long-term tracking.




## Results

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOT<br>PNorm (%)</th>
    <th>LaSOT<br>P (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>TrackingNet<br>PNorm (%)</th>

    <th>OTB100<br>AUC (%)</th>
    <th>UAV123<br>AUC (%)</th>

  </tr>
  <tr>
    <td>CTT</td>
    <td>65.7</td>
    <td>75.0</td>
    <td>69.8</td>
    <td>81.4</td>
    <td>86.4</td>
    <td>70.1</td>
    <td>68.6</td>
  </tr>

</table>

## Installation
Installation for our project is same with [TransT](https://github.com/chenxin-dlut/TransT)

## Citation

```
@inproceedings{zhong2022correlation,
  title={Correlation-Based Transformer Tracking},
  author={Zhong, Minghan and Chen, Fanglin and Xu, Jun and Lu, Guangming},
  booktitle={Artificial Neural Networks and Machine Learning--ICANN 2022: 31st International Conference on Artificial Neural Networks, Bristol, UK, September 6--9, 2022, Proceedings, Part I},
  pages={85--96},
  year={2022},
  organization={Springer}
}
```  

## Acknowledgement
This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking) based on **Pytorch** and [TransT](https://github.com/chenxin-dlut/TransT). 
We would like to thank their authors for providing great frameworks and toolkits.



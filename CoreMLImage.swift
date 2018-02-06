//
//  Created by Daryl Rowland on 15/12/2017.
//  Copyright © Jigsaw XYZ
//

import Foundation
import UIKit
import CoreML
import AVFoundation
import Vision

@available(iOS 11.0, *)
@objc(CoreMLImage)
public class CoreMLImage: UIView, AVCaptureVideoDataOutputSampleBufferDelegate {

  var bridge: RCTEventDispatcher!
  var captureSession: AVCaptureSession?
  var videoPreviewLayer: AVCaptureVideoPreviewLayer?
  var model: VNCoreMLModel?
  var lastClassification: String = ""
  var onClassification: RCTBubblingEventBlock?

  required public init(coder aDecoder: NSCoder) {
    super.init(coder: aDecoder)!
  }

  override init(frame: CGRect) {
    super.init(frame: frame)
    self.frame = frame;
  }

  func capture(_ captureOutput: AVCaptureFileOutput!, didStartRecordingToOutputFileAt fileURL: URL!, fromConnections connections: [Any]!) {
  }

  public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    let img = self.imageFromSampleBuffer(sampleBuffer: sampleBuffer)
    runMachineLearning(img: img)
  }

  func runMachineLearning(img: CIImage) {
    let image = self.crop(image: self.convertToUIImage(ciimage: img), withWidth: 224, andHeight: 224)
    let ciImage = CIImage(image: image!)
    if (self.model != nil) {
      let request = VNCoreMLRequest(model: self.model!, completionHandler: { [weak self] request, error in
        self?.processClassifications(for: request, error: error)
      })
      request.imageCropAndScaleOption = .centerCrop

      let orientation = CGImagePropertyOrientation.up

      DispatchQueue.global(qos: .userInitiated).async {
        let handler = VNImageRequestHandler(ciImage: ciImage!, orientation: orientation)
        do {
          try handler.perform([request])
        } catch {
          print("Failed to perform classification.\n\(error.localizedDescription)")
        }
      }
    }
  }

  func processClassifications(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      guard let results = request.results else {
        print("Unable to classify image")
        print(error!.localizedDescription)
        return
      }
      let imageMultiArray = (results[0] as AnyObject).featureValue.multiArrayValue
      var vector = [NSNumber]()
      for i in 0..<((imageMultiArray?.count)!) {
        vector.append(imageMultiArray![i])
      }
      self.onClassification!(["vector": vector])
    }
  }

  func convertToUIImage(ciimage: CIImage) -> UIImage {
    let context:CIContext = CIContext.init(options: nil)
    let cgImage:CGImage = context.createCGImage(ciimage, from: ciimage.extent)!
    let image:UIImage = UIImage.init(cgImage: cgImage)
    return image
  }

  func crop(image: UIImage, withWidth width: Double, andHeight height: Double) -> UIImage? {
    if let cgImage = image.cgImage {
      let contextImage: UIImage = UIImage(cgImage: cgImage)
      let contextSize: CGSize = contextImage.size
      var posX: CGFloat = 0.0
      var posY: CGFloat = 0.0
      var cgwidth: CGFloat = CGFloat(width)
      var cgheight: CGFloat = CGFloat(height)

      if contextSize.width > contextSize.height {
        posX = ((contextSize.width - contextSize.height) / 2)
        posY = 0
        cgwidth = contextSize.height
        cgheight = contextSize.height
      } else {
        posX = 0
        posY = ((contextSize.height - contextSize.width) / 2)
        cgwidth = contextSize.width
        cgheight = contextSize.width
      }

      let rect: CGRect = CGRect(x: posX, y: posY, width: cgwidth, height: cgheight)

      var croppedContextImage: CGImage? = nil
      if let contextImage = contextImage.cgImage {
        if let croppedImage = contextImage.cropping(to: rect) {
          croppedContextImage = croppedImage
        }
      }

      if let croppedImage:CGImage = croppedContextImage {
        let image: UIImage = UIImage(cgImage: croppedImage, scale: image.scale, orientation: image.imageOrientation)
        return image
      }
    }
    return nil
  }

  func imageFromSampleBuffer(sampleBuffer : CMSampleBuffer) -> CIImage
  {
    let imageBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
    let ciimage : CIImage = CIImage(cvPixelBuffer: imageBuffer)
    return ciimage
  }

  override public func layoutSubviews() {
    super.layoutSubviews()
    let view = UIView(frame: CGRect(x: 0, y: 0, width: self.frame.width, height: self.frame.height))
    let captureDevice = AVCaptureDevice.default(for: .video)

    do {
      if (captureDevice != nil) {
        let input = try AVCaptureDeviceInput(device: captureDevice!)
        self.captureSession = AVCaptureSession()
        self.captureSession?.addInput(input)

        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession!)
        videoPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        videoPreviewLayer?.frame = view.layer.bounds

        view.layer.addSublayer(videoPreviewLayer!)
        self.addSubview(view)

        let videoDataOutput = AVCaptureVideoDataOutput()
        let queue = DispatchQueue(label: "xyz.jigswaw.ml.queue")
        videoDataOutput.setSampleBufferDelegate(self, queue: queue)
        guard (self.captureSession?.canAddOutput(videoDataOutput))! else {
          fatalError()
        }
        self.captureSession?.addOutput(videoDataOutput)
        self.captureSession?.startRunning()
      }

    } catch {
      print(error)
    }

  }

  @objc(setModelFile:) public func setModelFile(modelFile: String) {
    print("Setting model file to: " + modelFile)
    let path = Bundle.main.url(forResource: modelFile, withExtension: "mlmodelc")

    do {
      let modelUrl = try MLModel.compileModel(at: path!)
      let model = try MLModel.init(contentsOf: modelUrl)
      self.model = try VNCoreMLModel(for: model)

    } catch {
      print("Error")
    }

  }

  @objc(setOnClassification:) public func setOnClassification(onClassification: @escaping RCTBubblingEventBlock) {
    self.onClassification = onClassification
  }

}

import Foundation
import AVFoundation

actor Recorder {
    private var recorder: AVAudioRecorder?
    
    enum RecorderError: Error {
        case couldNotStartRecording
    }
    
    func startRecording(toOutputFile url: URL, delegate: AVAudioRecorderDelegate?) throws {
        let recordSettings: [String : Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
#if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default)
#endif
        let recorder = try AVAudioRecorder(url: url, settings: recordSettings)
        recorder.delegate = delegate
        if recorder.record() == false {
            print("Could not start recording")
            throw RecorderError.couldNotStartRecording
        }
        self.recorder = recorder
    }
    
    func stopRecording() {
        recorder?.stop()
        recorder = nil
    }
}

final class RealtimeRecorder {
    private let sampleRate = 16000.0
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var onSamples: (([Float]) -> Void)?

    enum RealtimeRecorderError: Error {
        case couldNotCreateFormat
        case couldNotCreateConverter
    }

    func start(onSamples: @escaping ([Float]) -> Void) throws {
        stop()

        self.onSamples = onSamples

#if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
#endif

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw RealtimeRecorderError.couldNotCreateFormat
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw RealtimeRecorderError.couldNotCreateConverter
        }

        self.targetFormat = targetFormat
        self.converter = converter

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.process(buffer)
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        if engine.isRunning {
            engine.stop()
        }
        engine.inputNode.removeTap(onBus: 0)
        converter = nil
        targetFormat = nil
        onSamples = nil

#if !os(macOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
#endif
    }

    private func process(_ buffer: AVAudioPCMBuffer) {
        guard let converter, let targetFormat else {
            return
        }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let frameCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCapacity) else {
            return
        }

        var didProvideBuffer = false
        var conversionError: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, outputStatus in
            if didProvideBuffer {
                outputStatus.pointee = .noDataNow
                return nil
            }

            didProvideBuffer = true
            outputStatus.pointee = .haveData
            return buffer
        }

        converter.convert(to: convertedBuffer, error: &conversionError, withInputFrom: inputBlock)

        guard conversionError == nil, let channelData = convertedBuffer.floatChannelData?[0] else {
            return
        }

        let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(convertedBuffer.frameLength)))
        onSamples?(samples)
    }
}

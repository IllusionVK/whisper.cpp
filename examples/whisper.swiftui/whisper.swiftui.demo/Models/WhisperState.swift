import Foundation
import SwiftUI
import AVFoundation

@MainActor
class WhisperState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canTranscribe = false
    @Published var isRecording = false
    @Published var isRealtimeTranscribing = false
    @Published var realtimeTranscript = ""
    
    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private let realtimeRecorder = RealtimeRecorder()
    private var recordedFile: URL? = nil
    private var audioPlayer: AVAudioPlayer?
    private var realtimeSamples: [Float] = []
    private var realtimeTask: Task<Void, Never>?
    private let realtimeSampleRate = 16000
    private let realtimeTranscribeInterval: UInt64 = 2_000_000_000
    private let realtimeWindowSampleCount = 16000 * 8
    private let realtimeMinimumSampleCount = 16000
    
    private var builtInModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin", subdirectory: "models")
    }
    
    private var sampleUrl: URL? {
        Bundle.main.url(forResource: "jfk", withExtension: "wav", subdirectory: "samples")
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    override init() {
        super.init()
        loadModel()
    }
    
    func loadModel(path: URL? = nil, log: Bool = true) {
        do {
            whisperContext = nil
            if (log) { messageLog += "Loading model...\n" }
            let modelUrl = path ?? builtInModelUrl
            if let modelUrl {
                whisperContext = try WhisperContext.createContext(path: modelUrl.path())
                if (log) { messageLog += "Loaded model \(modelUrl.lastPathComponent)\n" }
            } else {
                if (log) { messageLog += "Could not locate model\n" }
            }
            canTranscribe = whisperContext != nil
        } catch {
            print(error.localizedDescription)
            if (log) { messageLog += "\(error.localizedDescription)\n" }
            canTranscribe = false
        }
    }

    func benchCurrentModel() async {
        if whisperContext == nil {
            messageLog += "Cannot bench without loaded model\n"
            return
        }
        messageLog += "Running benchmark for loaded model\n"
        let result = await whisperContext?.benchFull(modelName: "<current>", nThreads: Int32(min(4, cpuCount())))
        if (result != nil) { messageLog += result! + "\n" }
    }

    func bench(models: [Model]) async {
        let nThreads = Int32(min(4, cpuCount()))

//        messageLog += "Running memcpy benchmark\n"
//        messageLog += await WhisperContext.benchMemcpy(nThreads: nThreads) + "\n"
//
//        messageLog += "Running ggml_mul_mat benchmark with \(nThreads) threads\n"
//        messageLog += await WhisperContext.benchGgmlMulMat(nThreads: nThreads) + "\n"

        messageLog += "Running benchmark for all downloaded models\n"
        messageLog += "| CPU | OS | Config | Model | Th | FA | Enc. | Dec. | Bch5 | PP | Commit |\n"
        messageLog += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        for model in models {
            loadModel(path: model.fileURL, log: false)
            if whisperContext == nil {
                messageLog += "Cannot bench without loaded model\n"
                break
            }
            let result = await whisperContext?.benchFull(modelName: model.name, nThreads: nThreads)
            if (result != nil) { messageLog += result! + "\n" }
        }
        messageLog += "Benchmarking completed\n"
    }
    
    func transcribeSample() async {
        if let sampleUrl {
            await transcribeAudio(sampleUrl)
        } else {
            messageLog += "Could not locate sample\n"
        }
    }
    
    private func transcribeAudio(_ url: URL) async {
        if (!canTranscribe) {
            return
        }
        guard let whisperContext else {
            return
        }
        
        do {
            canTranscribe = false
            messageLog += "Reading wave samples...\n"
            let data = try readAudioSamples(url)
            messageLog += "Transcribing data...\n"
            await whisperContext.fullTranscribe(samples: data)
            let text = await whisperContext.getTranscription()
            messageLog += "Done: \(text)\n"
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)\n"
        }
        
        canTranscribe = true
    }
    
    private func readAudioSamples(_ url: URL) throws -> [Float] {
        stopPlayback()
        try startPlayback(url)
        return try decodeWaveFile(url)
    }
    
    func toggleRecord() async {
        if isRecording {
            await recorder.stopRecording()
            isRecording = false
            if let recordedFile {
                await transcribeAudio(recordedFile)
            }
        } else {
            requestRecordPermission { granted in
                if granted {
                    Task { @MainActor in
                        do {
                            self.stopPlayback()
                            let file = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                                .appending(path: "output.wav")
                            try await self.recorder.startRecording(toOutputFile: file, delegate: self)
                            self.isRecording = true
                            self.recordedFile = file
                        } catch {
                            print(error.localizedDescription)
                            self.messageLog += "\(error.localizedDescription)\n"
                            self.isRecording = false
                        }
                    }
                }
            }
        }
    }

    func toggleRealtimeTranscription() async {
        if isRealtimeTranscribing {
            stopRealtimeTranscription()
        } else {
            startRealtimeTranscription()
        }
    }

    private func startRealtimeTranscription() {
        guard canTranscribe else {
            return
        }

        guard whisperContext != nil else {
            messageLog += "Cannot transcribe without loaded model\n"
            return
        }

        requestRecordPermission { granted in
            if granted {
                Task { @MainActor in
                    do {
                        self.stopPlayback()
                        self.realtimeSamples.removeAll(keepingCapacity: true)
                        self.realtimeTranscript = ""

                        try self.realtimeRecorder.start { [weak self] samples in
                            Task { @MainActor in
                                self?.appendRealtimeSamples(samples)
                            }
                        }

                        self.isRealtimeTranscribing = true
                        self.canTranscribe = false
                        self.messageLog += "Live transcription started\n"
                        self.realtimeTask = Task { [weak self] in
                            await self?.runRealtimeTranscriptionLoop()
                        }
                    } catch {
                        print(error.localizedDescription)
                        self.messageLog += "\(error.localizedDescription)\n"
                        self.stopRealtimeTranscription(log: false)
                    }
                }
            } else {
                Task { @MainActor in
                    self.messageLog += "Record permission denied\n"
                }
            }
        }
    }

    private func stopRealtimeTranscription(log: Bool = true) {
        realtimeTask?.cancel()
        realtimeTask = nil
        realtimeRecorder.stop()
        realtimeSamples.removeAll(keepingCapacity: true)
        isRealtimeTranscribing = false
        canTranscribe = whisperContext != nil

        if log {
            messageLog += "Live transcription stopped\n"
        }
    }

    private func appendRealtimeSamples(_ samples: [Float]) {
        guard isRealtimeTranscribing else {
            return
        }

        realtimeSamples.append(contentsOf: samples)

        let maximumSampleCount = realtimeSampleRate * 30
        if realtimeSamples.count > maximumSampleCount {
            realtimeSamples.removeFirst(realtimeSamples.count - maximumSampleCount)
        }
    }

    private func runRealtimeTranscriptionLoop() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(nanoseconds: realtimeTranscribeInterval)
            } catch {
                break
            }

            await transcribeRealtimeBuffer()
        }
    }

    private func transcribeRealtimeBuffer() async {
        guard isRealtimeTranscribing, let whisperContext else {
            return
        }

        guard realtimeSamples.count >= realtimeMinimumSampleCount else {
            return
        }

        let samples = Array(realtimeSamples.suffix(realtimeWindowSampleCount))
        await whisperContext.fullTranscribe(samples: samples)
        let text = (await whisperContext.getTranscription()).trimmingCharacters(in: .whitespacesAndNewlines)

        guard isRealtimeTranscribing, !Task.isCancelled else {
            return
        }

        if !text.isEmpty {
            mergeRealtimeTranscript(text)
        }
    }

    private func mergeRealtimeTranscript(_ latestText: String) {
        let latestWords = latestText.split(whereSeparator: \.isWhitespace)
        guard !latestWords.isEmpty else {
            return
        }

        if realtimeTranscript.isEmpty {
            realtimeTranscript = latestWords.joined(separator: " ")
            return
        }

        let currentWords = realtimeTranscript.split(whereSeparator: \.isWhitespace)
        let overlap = largestTranscriptOverlap(currentWords: currentWords, latestWords: latestWords)
        let addition = latestWords.dropFirst(overlap).joined(separator: " ")
        guard !addition.isEmpty else {
            return
        }

        realtimeTranscript += " " + addition
    }

    private func largestTranscriptOverlap(currentWords: [Substring], latestWords: [Substring]) -> Int {
        let maximumOverlap = min(currentWords.count, latestWords.count)
        guard maximumOverlap > 0 else {
            return 0
        }

        for overlap in stride(from: maximumOverlap, through: 1, by: -1) {
            let currentSuffix = currentWords.suffix(overlap).map(normalizedTranscriptWord)
            let latestPrefix = latestWords.prefix(overlap).map(normalizedTranscriptWord)
            if currentSuffix == latestPrefix {
                return overlap
            }
        }

        return 0
    }

    private func normalizedTranscriptWord(_ word: Substring) -> String {
        word.lowercased().trimmingCharacters(in: .punctuationCharacters)
    }
    
    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
#if os(macOS)
        response(true)
#else
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            response(granted)
        }
#endif
    }
    
    private func startPlayback(_ url: URL) throws {
        audioPlayer = try AVAudioPlayer(contentsOf: url)
        audioPlayer?.play()
    }
    
    private func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }
    
    // MARK: AVAudioRecorderDelegate
    
    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        if let error {
            Task {
                await handleRecError(error)
            }
        }
    }
    
    private func handleRecError(_ error: Error) {
        print(error.localizedDescription)
        messageLog += "\(error.localizedDescription)\n"
        isRecording = false
    }
    
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task {
            await onDidFinishRecording()
        }
    }
    
    private func onDidFinishRecording() {
        isRecording = false
    }
}


fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}

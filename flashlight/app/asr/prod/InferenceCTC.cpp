/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/decoder/DecodeUtils.h"
#include "flashlight/app/asr/decoder/Defines.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "jsonbuilder/include/jsonbuilder/JsonBuilder.h"
#include "jsonbuilder/include/jsonbuilder/JsonRenderer.h"

//<======================================== Member functions ========================================>
void printArray(af::array arr,std::string name, bool printdata){
  std::cout<<"Dimensions of the "<<name<<" are "<<arr.dims(0);
  for(int i =1;i<arr.numdims();i++){
    std::cout<<"X"<<arr.dims(i);
  }
  std::cout<<std::endl;
  std::cout<<"Dim4 dimensions are : "<<arr.dims().dims[0]<<"X"<<arr.dims().dims[1]<<"X"<<arr.dims().dims[2]<<"X"<<arr.dims().dims[3]<<std::endl;
  std::cout<<std::endl;
  if(printdata){
    af_print_array(arr.get());
  }
  std::cout<<std::endl;
}

DEFINE_string(
    am_path,
    "",
    "Path to CTC trained acousitc mode to perform inference");
DEFINE_string(tokens_path, "", "Path to the model tokens set");
DEFINE_string(
    lexicon_path,
    "",
    "Path to the lexicon which defines mapping between word and tokens + restricts beam search");
DEFINE_string(
    lm_path,
    "",
    "Path to ngram language model. Either arpa file or KenLM bin file");
DEFINE_int32(beam_size, 100, "Beam size for the beam-search decoding");
DEFINE_int32(
    beam_size_token,
    10,
    "Tokens beam size for the beam-search decoding");
DEFINE_double(beam_threshold, 100, "Beam-search decoding pruning parameters");
DEFINE_double(lm_weight, 3, "Beam-search decoding language model weight");
DEFINE_double(word_score, 0, "Beam-search decoding word addition score");
DEFINE_int64(sample_rate, 16000, "Sample rate of the input audio");
DEFINE_string(
    audio_list,
    "",
    "Path to the file where each row is audio file path, "
    "if it is empty interactive regime is used");

void serializeAndCheckFlags() {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  std::string currVal;
  gflags::GetAllFlags(&allFlags);
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << ";";
  }
  LOG(INFO) << "Gflags after parsing\n" << serialized.str();

  std::unordered_map<std::string, std::string> flgs = {
      {"am_path", FLAGS_am_path},
      {"tokens_path", FLAGS_tokens_path},
      {"lexicon_path", FLAGS_lexicon_path},
      {"lm_path", FLAGS_lm_path}};
  for (auto& path : flgs) {
    if (path.second.empty() || !fl::lib::fileExists(path.second)) {
      throw std::runtime_error(
          "[Inference tutorial for CTC] Invalid file path specified for the flag --" +
          path.first + " with value '" + path.second +
          "': either it is empty or doesn't exist.");
    }
  }
}

void loadModel(
    std::shared_ptr<fl::Module>& network,
    std::unordered_map<std::string, std::string>& networkFlags,
    std::shared_ptr<fl::app::asr::SequenceCriterion>& criterion) {
  std::unordered_map<std::string, std::string> cfg;
  std::string version;

  LOG(INFO) << "[Inference tutorial for CTC] Reading acoustic model from "
            << FLAGS_am_path;
  fl::setDevice(0);
  fl::ext::Serializer::load(FLAGS_am_path, version, cfg, network, criterion);
  if (version != FL_APP_ASR_VERSION) {
    LOG(WARNING) << "[Inference tutorial for CTC] Acostuc model version "
                 << version << " and code version " << FL_APP_ASR_VERSION;
  }
  if (cfg.find(fl::app::asr::kGflags) == cfg.end()) {
    LOG(FATAL)
        << "[Inference tutorial for CTC] Invalid config is loaded from acoustic model"
        << FLAGS_am_path;
  }
  for (auto line : fl::lib::split("\n", cfg[fl::app::asr::kGflags])) {
    if (line == "") {
      continue;
    }
    auto res = fl::lib::split("=", line);
    if (res.size() >= 2) {
      auto key = fl::lib::split("--", res[0])[1];
      networkFlags[key] = res[1];
    }
  }
  if (networkFlags["criterion"] != fl::app::asr::kCtcCriterion) {
    LOG(FATAL)
        << "[Inference tutorial for CTC]: provided model is trained not with CTC, but with "
        << networkFlags["criterion"]
        << ". This type is not supported in the tutorial";
  }
}

class Inference {
  public:
    std::shared_ptr<fl::Module> network;
    std::unordered_map<std::string, std::string> networkFlags;
    std::shared_ptr<fl::app::asr::SequenceCriterion> criterion;
    fl::lib::text::Dictionary tokenDict;
    fl::lib::text::LexiconMap lexicon;
    fl::lib::text::Dictionary wordDict;
    std::shared_ptr<fl::lib::text::KenLM> lm;
    fl::lib::text::LexiconDecoder *decoder;
    fl::Dataset::DataTransformFunction inputTransform;
    std::string transcript;
    int unkWordIdx;

    Inference(
      std::shared_ptr<fl::Module>& network,
      std::unordered_map<std::string, std::string>& networkFlags,
      std::shared_ptr<fl::app::asr::SequenceCriterion>& criterion,
      fl::lib::text::Dictionary& tokenDict,
      fl::lib::text::LexiconMap& lexicon,
      fl::lib::text::Dictionary& wordDict,
      std::shared_ptr<fl::lib::text::KenLM>& lm,
      fl::lib::text::LexiconDecoder *decoder,
      fl::Dataset::DataTransformFunction& inputTransform,
      std::string& transcript,
      int unkWordIdx
    ) {
      this->network = network;
      this->decoder = decoder;
      this->networkFlags = networkFlags;
      this->tokenDict = tokenDict;
      this->lexicon = lexicon;
      this->wordDict = wordDict;
      this->lm = lm;
      this->inputTransform = inputTransform;
      this->criterion = criterion;
      this->transcript = transcript;
      this->unkWordIdx = unkWordIdx;
    }

    fl::Variable Forward(fl::app::asr::SoundInfo& audioInfo, std::vector<float>& audio) {
      std::cout<< "audio Info Details: channels = "<<audioInfo.channels<<" samplerate = "<<audioInfo.samplerate<<" frames = "<<audioInfo.frames<<std::endl;
      std::cout<< "audio size = "<<audio.size()<<"frames\n";
      std::cout<<"audio duration = "<<audioInfo.frames/audioInfo.samplerate<<std::endl;
      af::array input = this->inputTransform(
          static_cast<void*>(audio.data()),
          af::dim4(audioInfo.channels, audioInfo.frames),
          af::dtype::f32);
      printArray(input,"input array ", false);
      auto inputLen = af::constant(input.dims(0), af::dim4(1));
      printArray(inputLen, "inputLen array ", false);
      auto rawEmission = fl::ext::forwardSequentialModuleWithPadMask(
         fl::input(input), this->network, inputLen);
      printArray(rawEmission.array(), " Raw Emission array ", true);

      return rawEmission;
    }

    std::string  Decode(int nframes, int ntokens, std::vector<float>& emission) {
      // actually decode is not needed for evaluation
      const auto& result = this->decoder->decode(
        emission.data(),
        nframes /* time */,
        ntokens /* ntokens */);


      std::cout<<"Result size = "<<result.size()<<std::endl;
      int i = 0;
      std::cout<<"Result vector containns....."<<std::endl;
      // PRINT RESULTS
      for(auto f:result){
        std::cout<<"For position "<<i<<std::endl;
        std::cout<<"Score = "<<f.score<<" am score = "<<f.amScore<<" lmscore = "<<f.lmScore<<std::endl;
        // std::cout<<"Words size = "<<f.words.size()<<" Words are ............\n";
        // for(auto word:f.words){
        //   std::cout<<word<<" ";
        // }
        std::cout<<std::endl<<"Tokens size = "<<f.tokens.size()<<" Tokens are..........."<<std::endl;
        for(auto token:f.tokens){
          if(tokenDict.getEntry(token)!="#")
          std::cout<<tokenDict.getEntry(token)<<" ";
        }
        std::cout<<std::endl;
        i++;
        if(i==10)
        break;
      }
      std::cout<<std::endl;
      // Take top hypothesis and cleanup predictions
      auto rawWordPrediction = result[0].words;
      auto rawTokenPrediction = result[0].tokens;

      auto tokenPrediction = fl::app::asr::tknPrediction2Ltr(
          rawTokenPrediction,
          this->tokenDict,
          fl::app::asr::kCtcCriterion,
          this->networkFlags["surround"],
          false /* eostoken */,
          0 /* replabel */,
          false /* usewordpiece */,
          this->networkFlags["wordseparator"]);
      rawWordPrediction =
          fl::app::asr::validateIdx(rawWordPrediction, this->unkWordIdx);
      auto wordPrediction = fl::app::asr::wrdIdx2Wrd(rawWordPrediction, wordDict);
      auto wordPredictionStr = fl::lib::join(" ", wordPrediction);

      LOG(INFO) << "[Inference tutorial for CTC]: predicted output: "
                << wordPredictionStr;

      return wordPredictionStr;
    }

    af::array ForceAlign(fl::Variable& rawEmission) {
      int i, j;
      std::vector<int> targetvec(this->transcript.size()+1);
      for(i=0; i<this->transcript.size(); i++) {
        char ch = this->transcript.at(i);
        if (ch==' ') ch = '|';
        targetvec[i] = this->tokenDict.getIndex(std::string(1, ch));
      }
      targetvec[i] = this->tokenDict.getIndex("|");

      auto target = af::array(targetvec.size(), 1, targetvec.data());
      std::cout << "target Dims " << target.dims(0) << " X " << target.dims(1) << " X " << target.dims(2) << " X " << target.dims(3) << " X " << std::endl;
      std::cout << "Now doing force alignment using output: " << this->transcript << std::endl;
      // Now doing force alignment and printing the softmax values
      // calculate softmax for rawEmission
      auto paths = this->criterion->viterbiPathWithTarget(
          rawEmission.array(), target);

      return paths;
    }

    std::vector<std::vector<float>> GetSoftMaxProbs(int nframes, int ntokens, std::vector<float>& emission) {
      std::cout << "Now calculating softmax from emissions" << std::endl;
// calculate the softmax probabilities assuming column major order of emission vector.
      std::vector<std::vector<float>> probs(ntokens, std::vector<float>(nframes, 0));
      
      int i, j;

      for(i=0; i<ntokens; i++) {
          std::cout << this->tokenDict.getEntry(i) << " ";
      }
      std::cout << std::endl;

      for(j=0; j<nframes; j++) {
        float s = 0;
        for(i=0; i<ntokens; i++) {
          probs[i][j] = exp(emission[j*ntokens+i]);
          s += probs[i][j];
        }
        int mindex = 0;
        for(i=0; i<ntokens; i++) {
          probs[i][j] /= s;
          if (probs[mindex][j] < probs[i][j]) mindex = i;
        }
        std::cout << "frame " << j << ": " << this->tokenDict.getEntry(mindex) << ":" << probs[mindex][j] << std::endl;
      }

      return probs;
    }

    const char* Evaluate(af::array paths, std::vector<std::vector<float>> probs, std::string transcript) {
      // bestPaths contain the index tokens
      // we need to return:
      // word start and end times
      // word score
      // overall score
      // return a JSON string from here.
      jsonbuilder::JsonBuilder jb;
      jsonbuilder::JsonIterator wordsIterator = jb.push_back(jb.end(), "words", jsonbuilder::JsonArray);
      
      const int B = paths.dims(1);
      const int T = paths.dims(0);
      std::cout << "Paths dims = " << B << "x" << T << std::endl;
      float score = 0;
      int len = 0;
      std::vector<std::string> tokens;
      bool inside_word = false;
      std::string cur_word = "";
      float word_score = 0;
      int word_len = 0;
      int word_start = 0;
      int word_end = 0;

      int t;

      for (t = 0; t < T; t++) {
        int p = paths(t, 0).scalar<int>();
        if (p == -1) {
          break;
        }
        auto token = tokenDict.getEntry(p);
        tokens.push_back(token);
        std::cout << t << ": " << token << std::endl;

        // if a word haas started
        if(!inside_word && p!=28 && p!=1) {
          inside_word = true;
          word_start = t;
          word_score = 0;
          word_len = 0;
          cur_word = "";
        }

        // if a wrd has ended
        if(inside_word && p==1) {
          inside_word = false;
          if (cur_word.back()=='#') {
            cur_word.pop_back();
          }
          word_score /= word_len;
          word_end = t-1;
          // add to json the word and the word score
          jsonbuilder::JsonIterator singlewordIterator = jb.push_back(wordsIterator, "", jsonbuilder::JsonObject);
          jb.push_back(singlewordIterator, "word", cur_word);
          jb.push_back(singlewordIterator, "start_time", word_start*30);
          jb.push_back(singlewordIterator, "end_time", word_end*30);
          jb.push_back(singlewordIterator, "score", word_score);
          std::cout << cur_word << " "  << word_score << std::endl;
        }

        // otherwise inside a word
        if(inside_word) {
          if(cur_word.size()==0 || cur_word.back()!=token.at(0)) {
            if (cur_word.back()=='#') {
              cur_word.back() = token.at(0);
            } else {
              cur_word += token.at(0);
            }
            if(p!=28) {
              word_score += probs[p][t];
              word_len++;
            }
          }
        }
        if(p!=28 && p!=1) {
          score += probs[p][t];
          len++;
        }
      }

      if(inside_word) {
        inside_word = false;
        if (cur_word.back()=='#') {
          cur_word.pop_back();
        }
        word_score /= word_len;
        word_end = t-1;
        // add to json the word and the word score
        jsonbuilder::JsonIterator singlewordIterator = jb.push_back(wordsIterator, "", jsonbuilder::JsonObject);
        jb.push_back(singlewordIterator, "word", cur_word);
        jb.push_back(singlewordIterator, "start_time", word_start*30);
        jb.push_back(singlewordIterator, "end_time", word_end*30);
        jb.push_back(singlewordIterator, "score", word_score);
        std::cout << cur_word << " "  << word_score << std::endl;
      }
      jb.push_back(jb.end(), "score", score/len);
      jb.push_back(jb.end(), "transcript", transcript);

      std::cout << "SCORE: "  << score/len << std::endl;

      jsonbuilder::JsonRenderer renderer;
      renderer.Reserve(2048);

      std::string_view result = renderer.Render(jb);
      std::string stl_string(result.data(), result.size());

      std::cout << stl_string.c_str() << std::endl;

      return stl_string.c_str();
    }


    void run(fl::app::asr::SoundInfo& audioInfo, std::vector<float>& audio) {
      // run forward pass
      fl::Variable rawEmission = this->Forward(audioInfo, audio);

      // prepare emission vector
      auto emission = fl::ext::afToVector<float>(rawEmission);
      std::cout<<"size of emission vector = "<<emission.size()<<std::endl;

      // calculate softmax from emissions for scoring later
      int nframes = rawEmission.dims(1); /* time */
      int ntokens = rawEmission.dims(0);
      auto probs = GetSoftMaxProbs(nframes, ntokens, emission);

      // decode the best outcome
      std::string decoded_text = Decode(nframes, ntokens, emission);

      // NOTE: if this>transcript is not there use decoded_text as transcript instead.

      // force align the intended transcript, if nothing than use current one
      auto alignedpath = ForceAlign(rawEmission);
      
      // evaluate the score and determing word boundaries
      auto jsonResponse = Evaluate(alignedpath, probs, decoded_text);

      std::cout << jsonResponse << std::endl;

      // add other info to the json response and finally return
      // return result

      // NOTE:- the actual words are word groups, so we may have to do some post alignment based upon speech hints as well
      // In the demo we will not do that though.
  }
};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }

  fl::init();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  serializeAndCheckFlags();
  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::unordered_map<std::string, std::string> networkFlags;
  std::shared_ptr<fl::app::asr::SequenceCriterion> criterion;
  loadModel(network, networkFlags, criterion);
  network->eval();
  criterion->eval();
  LOG(INFO) << "[Inference tutorial for CTC] Network is loaded.";
  /* ===================== Set All Dictionaries ===================== */
  fl::lib::text::Dictionary tokenDict(FLAGS_tokens_path);
  tokenDict.addEntry(fl::app::asr::kBlankToken);
  int blankIdx = tokenDict.getIndex(fl::app::asr::kBlankToken);
  int wordSepIdx = networkFlags["wordseparator"] == ""
      ? -1
      : tokenDict.getIndex(networkFlags["wordseparator"]);

  fl::lib::text::LexiconMap lexicon =
      fl::lib::text::loadWords(FLAGS_lexicon_path, -1);
  fl::lib::text::Dictionary wordDict = fl::lib::text::createWordDict(lexicon);
  LOG(INFO)
      << "[Inference tutorial for CTC] Number of classes/tokens in the network: "
      << tokenDict.indexSize();
  LOG(INFO) << "[Inference tutorial for CTC] Number of words in the lexicon: "
            << wordDict.indexSize();
  fl::lib::text::DictionaryMap dicts = {{0, tokenDict}, {1, wordDict}};
  /* ===================== Set LM, Trie, Decoder ===================== */
  int unkWordIdx = wordDict.getIndex(fl::lib::text::kUnkToken);
  auto lm = std::make_shared<fl::lib::text::KenLM>(FLAGS_lm_path, wordDict);
  if (!lm) {
    LOG(FATAL)
        << "[Inference tutorial for CTC] Only KenLM model for language model is supported. "
        << "Failed to load kenlm LM: " << FLAGS_lm_path;
  }
  LOG(INFO) << "[Inference tutorial for CTC] Language model is constructed.";
  std::shared_ptr<fl::lib::text::Trie> trie = fl::app::asr::buildTrie(
      "wrd" /* decoderType */,
      true /* useLexicon */,
      lm,
      "max" /* smearing */,
      tokenDict,
      lexicon,
      wordDict,
      wordSepIdx,
      0 /* repLabel */);
  LOG(INFO) << "[Inference tutorial for CTC] Trie is planted.";

  auto decoder = fl::lib::text::LexiconDecoder(
      {.beamSize = FLAGS_beam_size,
       .beamSizeToken = FLAGS_beam_size_token,
       .beamThreshold = FLAGS_beam_threshold,
       .lmWeight = FLAGS_lm_weight,
       .wordScore = FLAGS_word_score,
       .unkScore = -std::numeric_limits<float>::infinity(),
       .silScore = 0,
       .logAdd = false,
       .criterionType = fl::lib::text::CriterionType::CTC},
      trie,
      lm,
      wordSepIdx,
      blankIdx,
      unkWordIdx,
      std::vector<float>(),
      false);
  LOG(INFO) << "[Inference tutorial for CTC] Beam search decoder is created";
  /* ===================== Audio Loading Preparation ===================== */
  fl::lib::audio::FeatureParams featParams(
      FLAGS_sample_rate,
      std::atoll(networkFlags["framesizems"].c_str()),
      std::atoll(networkFlags["framestridems"].c_str()),
      std::atoll(networkFlags["filterbanks"].c_str()),
      std::atoll(networkFlags["lowfreqfilterbank"].c_str()),
      std::atoll(networkFlags["highfreqfilterbank"].c_str()),
      std::atoll(networkFlags["mfcccoeffs"].c_str()),
      fl::app::asr::kLifterParam /* lifterparam */,
      std::atoll(networkFlags["devwin"].c_str()) /* delta window */,
      std::atoll(networkFlags["devwin"].c_str()) /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  fl::app::asr::FeatureType featType;
  if (networkFlags.find("features_type") != networkFlags.end()) {
    featType = fl::app::asr::getFeatureType(
                   networkFlags["features_type"], 1, featParams)
                   .second;
  } else {
    // old models TODO remove as per @avidov converting scirpt
    if (networkFlags["pow"] == "true") {
      featType = fl::app::asr::FeatureType::POW_SPECTRUM;
    } else if (networkFlags["mfsc"] == "true") {
      featType = fl::app::asr::FeatureType::MFSC;
    } else if (networkFlags["mfcc"] == "true") {
      featType = fl::app::asr::FeatureType::MFCC;
    } else {
      // raw wave
      featType = fl::app::asr::FeatureType::NONE;
    }
  }
  auto inputTransform = fl::app::asr::inputFeatures(
      featParams,
      featType,
      {networkFlags["localnrmlleftctx"] == "true",
       networkFlags["localnrmlrightctx"] == "true"},
      /*sfxConf=*/{});
  fl::EditDistanceMeter dst;

  /* ===================== Inference ===================== */
  bool interactive = FLAGS_audio_list == "";
  std::ifstream audioListStream;
  if (!interactive) {
    audioListStream = std::ifstream(FLAGS_audio_list);
  }

  std::string transcript = "hello world";
  // create an inference engine
  Inference inferEngine(network,networkFlags,criterion,tokenDict,lexicon,wordDict,lm,&decoder,inputTransform,transcript, unkWordIdx);
  
  while (true) {
    std::string audioPath;
    // need to work out a socket here
    if (interactive) {
      LOG(INFO)
          << "[Inference tutorial for CTC]: Waiting the input in the format [audio_path].";
      std::getline(std::cin, audioPath);
    } else {
      if (!std::getline(audioListStream, audioPath)) {
        return 0;
      }
    }
    if (audioPath == "") {
      LOG(INFO)
          << "[Inference tutorial for CTC]: Please provide non-empty input";
      continue;
    }
    if (!fl::lib::fileExists(audioPath)) {
      LOG(INFO) << "[Inference tutorial for CTC]: File '" << audioPath
                << "' doesn't exist, please provide valid audio path";
      continue;
    }
    auto audioInfo = fl::app::asr::loadSoundInfo(audioPath.c_str());
    auto audio = fl::app::asr::loadSound<float>(audioPath.c_str());
    std::cout<< "While doing inference for " << audioPath << " file\n";
    inferEngine.run(audioInfo, audio);
    
  }
  return 0;
}
/*
 Navicat Premium Data Transfer

 Source Server         : 本地链接
 Source Server Type    : MySQL
 Source Server Version : 80026
 Source Host           : localhost:3306
 Source Schema         : predict_log

 Target Server Type    : MySQL
 Target Server Version : 80026
 File Encoding         : 65001

 Date: 31/01/2023 18:44:56
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for data_model_2
-- ----------------------------
DROP TABLE IF EXISTS `data_model_2`;
CREATE TABLE `data_model_2`  (
  `seq` int unsigned NOT NULL,
  `methodSeq` int(0) NOT NULL,
  `context` mediumblob,
  `contextChanged` mediumblob,
  `parentId` int(0) DEFAULT NULL,
  `logNum` int(0) DEFAULT NULL,
  `logType` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `vectorStruct` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `vectorSemantic` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `leaf` varchar(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `fromLine` int(0) DEFAULT NULL,
  `toLine` int(0) DEFAULT NULL,
  `upperTreeBlockId` int(0) DEFAULT NULL,
  `deep` int(0) DEFAULT NULL,
  `syntacticMessage` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `lables` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  PRIMARY KEY (`seq`) USING BTREE,
  INDEX `data_model_2_methodSeq`(`methodSeq`) USING BTREE,
  INDEX `data_model_2_index_2`(`methodSeq`, `parentId`) USING BTREE,
  INDEX `data_model_2_index_1`(`methodSeq`, `leaf`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1812814 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
